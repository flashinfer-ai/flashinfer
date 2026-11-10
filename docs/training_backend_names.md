# CAKE training backend names

`det` identifies run-to-run bitwise determinism of the delivered forward and
backward for fixed inputs, hardware, execution plan and initial accumulation
buffers. It does not promise identical bits across shapes, devices, reduction
plans or parallel partitions. `kcp` is the suffix for a context-parallel route.

The rules and the target table below are the naming scheme for every CAKE
training family (FlashInfer issue #8928). Families move to it one per PR; until
a family's rename lands, the current names listed in its section below stay in
effect. `nondet` marks a package whose forward or backward does not promise
run-to-run bitwise results.

## Naming rules

1. **Module:** `flashinfer.cake_<op>_<mode>_train[_kcp]`. `mode` is `det` or `nondet`.
   - The public facade module always sits at the top level of `flashinfer`.
   - The backend package always sits at `flashinfer/experimental/cake/<module>/`.
   - Exception: a `_kcp` route may live in the backend package of its single-rank module as a submodule (`kcp.py`, `kcp_jit.py`). This applies when the route runs the single-rank programs and adds only its own boundary work. The route's libraries are then listed in that package's `build.json`. The `_kcp` facade still sits at the top level and imports the submodule lazily. Today only causal conv1d uses this form: its KCP route is the single-rank convolution plus a halo exchange, and a separate package would have to either copy the single-rank programs or import across packages.
2. **Autograd functional:** `cake_<op>_<mode>[_kcp]`.
3. **Autograd class:** `Cake<Op><Mode>[Kcp]Function`.
4. **Other functions:** the module name plus `_<verb>`, for example `_forward`, `_backward`, `_forward_workspace_bytes`, `_partition` or `_build_context`.
   - Use `forward` / `backward`, never `fwd` / `bwd`.
5. **Support query:** `is_<module>_supported`.
6. **Types:** `Cake<Op><Mode>[Kcp]<Noun>`, for example `CakeKDADetKcpContext`.
7. **Mode is part of the name, and only of the name:**
   - No entry point takes `deterministic=`.
   - No public name lacks a mode word. This includes families that have only one implementation today.
   - Module constants in ALL_CAPS (for example `ARCHES` or `SUPPORTED_COMPUTE_CAPABILITIES`) carry no mode word: the module path already qualifies them. Every other public name, including helper functions, classes and types, carries the mode word.
8. **`torch.use_deterministic_algorithms(True)`:**
   - A `nondet` entry raises PyTorch's alert at its non-deterministic step. With `warn_only=True` it warns and runs `nondet`. It never falls back to `det`.
   - `det` entries never alert.
9. **No aliases or shims for the old names.** Old names fail at import.
10. **Order of the name parts:** `op` → `mode` → `train` → `kcp` → verb.
    - The mode is the primary choice: it selects the package and the kernels. KCP is a variant within each mode.
    - All entries of one package share one prefix, such as `cake_kda_det_train_kcp_`.

## Target names

Module names are under `flashinfer.`. "Planned" marks versions that do not exist yet and will be added later under the same names.

| Family | Mode | Route | Module | Autograd entry | Autograd class | Current name |
|---|---|---|---|---|---|---|
| RMSNorm | det | single rank | `cake_rmsnorm_det_train` | `cake_rmsnorm_det` | `CakeRMSNormDetFunction` | unchanged |
| | nondet | single rank | `cake_rmsnorm_nondet_train` | `cake_rmsnorm_nondet` | `CakeRMSNormNondetFunction` | `cake_rmsnorm_train` / `cake_rmsnorm` |
| SiLU-gated RMSNorm | det | single rank | `cake_rmsnorm_silu_gate_det_train` | `cake_rmsnorm_silu_gate_det` | `CakeRMSNormSiLUGateDetFunction` | unchanged |
| | nondet | single rank | `cake_rmsnorm_silu_gate_nondet_train` | `cake_rmsnorm_silu_gate_nondet` | `CakeRMSNormSiLUGateNondetFunction` | `cake_rmsnorm_silu_gate_train` / `cake_rmsnorm_silu_gate` |
| causal conv1d | det | single rank | `cake_causal_conv1d_det_train` | `cake_causal_conv1d_det` | `CakeCausalConv1dDetFunction` | unchanged |
| | det | KCP | `cake_causal_conv1d_det_train_kcp` | `cake_causal_conv1d_det_kcp`, `cake_causal_conv1d_det_kcp_multi` | `CakeCausalConv1dDetKcpFunction`, `CakeCausalConv1dDetKcpMultiFunction` | `cake_causal_conv1d_det_kcp`, `cake_causal_conv1d_det_multi_kcp`, `CakeCausalConv1dDetFunctionKcp` |
| | nondet | single rank | `cake_causal_conv1d_nondet_train` | `cake_causal_conv1d_nondet` | `CakeCausalConv1dNondetFunction` | `experimental.cake.cake_causal_conv1d_train` / `cake_causal_conv1d` |
| | nondet | KCP | `cake_causal_conv1d_nondet_train_kcp` | `cake_causal_conv1d_nondet_kcp`, `cake_causal_conv1d_nondet_kcp_multi` | `CakeCausalConv1dNondetKcpFunction`, `CakeCausalConv1dNondetKcpMultiFunction` | `cake_causal_conv1d_train_kcp` / `cake_causal_conv1d_kcp`, `cake_causal_conv1d_multi_kcp` |
| mHC | det | single rank | `cake_mhc_det_train` | `cake_mhc_det_train_prepare`, `cake_mhc_det_train_finish` | `CakeMHCDetPrepareFunction`, `CakeMHCDetFinishFunction` | functions unchanged; classes `CakeMHCDetTrain*Function` |
| | nondet | single rank | `cake_mhc_nondet_train` | `cake_mhc_nondet_train_prepare`, `cake_mhc_nondet_train_finish` | `CakeMHCNondetPrepareFunction`, `CakeMHCNondetFinishFunction` | `experimental.cake.cake_mhc_train` / `cake_mhc_train_*` |
| mHC head | det | single rank | `cake_mhc_head_det_train` | `cake_mhc_head_det` | `CakeMHCHeadDetFunction` | `cake_mhc_det_train_head` / `CakeMHCDetTrainHeadFunction` |
| | nondet | single rank | `cake_mhc_head_nondet_train` | `cake_mhc_head_nondet` | `CakeMHCHeadNondetFunction` | new; no former name |
| GDN | det | single rank | `cake_gdn_det_train` (new top-level facade) | `cake_gdn_det` | `CakeGDNDetFunction` | `experimental.cake_gdn_det_train` / `chunk_gated_delta_rule_det` |
| | det | KCP | `cake_gdn_det_train_kcp` | `cake_gdn_det_kcp` | `CakeGDNDetKcpFunction` | `chunk_gdn_det_kcp` / `ChunkGDNDetFunctionKcp` |
| | nondet | single rank | `cake_gdn_nondet_train` | `cake_gdn_nondet` | `CakeGDNNondetFunction` | `experimental.cake.cake_gdn_train` / `chunk_gated_delta_rule(deterministic=False)` |
| | nondet | KCP | `cake_gdn_nondet_train_kcp` | `cake_gdn_nondet_kcp` | `CakeGDNNondetKcpFunction` | `cake_gdn_train_kcp` / `chunk_gdn_kcp` |
| KDA | det | single rank | `cake_kda_det_train` | `cake_kda_det` | `CakeKDADetFunction` | `chunk_kda_det` / `ChunkKDADetFunction` |
| | det | KCP | `cake_kda_det_train_kcp` | `cake_kda_det_kcp` | `CakeKDADetKcpFunction` | `chunk_kda_det_kcp` / `ChunkKDADetFunctionKcp` |
| | nondet | single rank | `cake_kda_nondet_train` | `cake_kda_nondet` | `CakeKDANondetFunction` | `cake_kda_train` selector and `experimental.cake.cake_kda_train` / `chunk_kda` |
| | nondet | KCP | `cake_kda_nondet_train_kcp` | `cake_kda_nondet_kcp` | `CakeKDANondetKcpFunction` | `cake_kda_train_kcp` / `chunk_kda_kcp` |
| LM head | det | single rank | `cake_lm_head_det_train` | `cake_lm_head_det_train_loss`, `cake_lm_head_det_train_logprob`, `cake_lm_head_det_train_support_logprob`, `cake_lm_head_det_train_support_kl` | per entry, by rule 3 | `chunked_lm_head_det_*` |
| | nondet | single rank | `cake_lm_head_nondet_train` | `cake_lm_head_nondet_train_loss` and the other three entries | per entry, by rule 3 | `chunked_lm_head_*(deterministic=False)` |
| MoE grouped GEMM | det | single rank | `cake_moe_grouped_gemm_det_train` | `cake_moe_grouped_gemm_det` | `CakeMoEGroupedGemmDetFunction` | `cake_grouped_det_mm` / `CakeGroupedDetMm` |
| | nondet | single rank | `cake_moe_grouped_gemm_nondet_train` | `cake_moe_grouped_gemm_nondet` | `CakeMoEGroupedGemmNondetFunction` | `cake_grouped_mm` |
| dense GEMM | det | single rank | `cake_dense_gemm_det_train` | `cake_dense_gemm_det_train_projection`, `cake_dense_gemm_det_train_router` | per entry, by rule 3 | `dense_projection_det_gemm`, `router_fp32_det_gemm` |
| | nondet | single rank | `cake_dense_gemm_nondet_train` | `cake_dense_gemm_nondet_train_projection`, `cake_dense_gemm_nondet_train_router` | per entry, by rule 3 | `experimental.dense_projection_gemm` `cake_backend` |
| DSA indexer | det | single rank | `cake_dsa_indexer_det_train` | `cake_dsa_indexer_det_train_topk`, `cake_dsa_indexer_det_train_topk_workspace_size` | none (no autograd: top-k selection) | `experimental.cake_dsa_indexer_det` (+ alias `experimental.cake_dsa_indexer`) / `dsa_indexer.dsa_indexer_det_topk(backend="cake")`, `dsa_indexer_det_topk_workspace_size`, aliases `dsa_indexer_topk*` |
| | nondet | single rank | `cake_dsa_indexer_nondet_train` | `cake_dsa_indexer_nondet_train_topk`, `cake_dsa_indexer_nondet_train_topk_workspace_size` | none (no autograd: top-k selection) | new; no former name |
| DSA sparse attention | det | single rank | `cake_dsa_sparse_attention_det_train` | `cake_dsa_sparse_attention_det`, `cake_dsa_sparse_attention_det_varlen`, `cake_dsa_sparse_attention_det_nope`, `cake_dsa_sparse_attention_det_nope_varlen` | `CakeDSASparseAttentionDetFunction` (both rope entries), `CakeDSASparseAttentionDetNopeFunction`, `CakeDSASparseAttentionDetNopeVarlenFunction` | new |
| | nondet | single rank | `cake_dsa_sparse_attention_nondet_train` | `cake_dsa_sparse_attention_nondet`, `cake_dsa_sparse_attention_nondet_varlen`, `cake_dsa_sparse_attention_nondet_nope`, `cake_dsa_sparse_attention_nondet_nope_varlen` | `CakeDSASparseAttentionNondetFunction` (both rope entries), `CakeDSASparseAttentionNondetNopeFunction`, `CakeDSASparseAttentionNondetNopeVarlenFunction` | `experimental.cake_dsa_train` / `dsa_sparse_attention.dsa_sparse_attention{,_varlen,_nope,_nope_varlen}` |
| GQA HD256 | det | single rank | `cake_gqa_det_train` | `cake_gqa_det` | `CakeGQADetFunction` | new; no former name |
| | nondet | single rank | `cake_gqa_nondet_train` | `cake_gqa_nondet` | `CakeGQANondetFunction` | `gqa_train` / `varlen_gqa_attention` |
| SwiGLU + MXFP8 quantization | det | single rank | `cake_gated_act_mxfp8_det_train` | `cake_gated_act_mxfp8_det_train_forward`, `cake_gated_act_mxfp8_det_train_backward` | none (no autograd: the forward and the backward are separate quantizing entries) | `gated_act_mxfp8` / `silu_and_mul_mxfp8_quantize`, `silu_and_mul_mxfp8_quantize_backward` |
| | nondet | single rank | | | | not planned: elementwise, no reduction to reorder |

### KCP entries: KDA det as the example

GDN and conv1d use the same pattern with their own stem. For `nondet`, replace `det` with `nondet`.

| Current | New |
|---|---|
| `chunk_kda_det_fwd_kcp` / `chunk_kda_det_bwd_kcp` | `cake_kda_det_train_kcp_forward` / `cake_kda_det_train_kcp_backward` |
| `partition_det_kcp` | `cake_kda_det_train_kcp_partition` |
| `build_context_det_kcp` | `cake_kda_det_train_kcp_build_context` |
| `kda_det_workspace_bytes_kcp` / `kda_det_saved_bytes_kcp` / `kda_det_comm_bytes_kcp` | `cake_kda_det_train_kcp_workspace_bytes` / `cake_kda_det_train_kcp_saved_bytes` / `cake_kda_det_train_kcp_comm_bytes` |
| `reduce_gate_param_grads_det_kcp` | `cake_kda_det_train_kcp_reduce_gate_param_grads` |
| `ContextDetKcp`, `PartitionDetKcp`, `WorkspaceDetKcp`, `SavedDetKcp`, `RecomputeHandleDetKcp` | `CakeKDADetKcpContext`, `CakeKDADetKcpPartition`, `CakeKDADetKcpWorkspace`, `CakeKDADetKcpSaved`, `CakeKDADetKcpRecomputeHandle` |

## Decisions

- **GDN `nondet` stays.** Today it is fixed-order in practice; its contract does not promise run-to-run bits.
- **mHC head becomes its own family.** It ships `det` and `nondet`. Like GDN `nondet`, the head's `nondet` is fixed-order (the same kernels as `det`); its contract does not promise run-to-run bits.
- **Causal conv1d KCP stays inside the single-rank backend packages.** `cake_causal_conv1d_det_train` and `cake_causal_conv1d_nondet_train` each carry a `kcp` submodule behind the top-level `cake_causal_conv1d_<mode>_train_kcp` facades (rule 1 exception).

## Current names

| Training family | Canonical backend / route name |
| --- | --- |
| LM head + loss | `cake_lm_head_det_train` (deterministic: module `flashinfer.cake_lm_head_det_train`, backend `flashinfer.experimental.cake.cake_lm_head_det_train`); `cake_lm_head_nondet_train` (non-deterministic: module `flashinfer.cake_lm_head_nondet_train`, backend `flashinfer.experimental.cake.cake_lm_head_nondet_train`) |

### LM head + loss

The deterministic module is `flashinfer.cake_lm_head_det_train`; its backend
package is `flashinfer.experimental.cake.cake_lm_head_det_train`. No entry takes
a `deterministic` keyword, and `torch.use_deterministic_algorithms(True)` changes
nothing. The former names are removed without aliases: importing one fails.

| Name | Former names (removed, no alias) |
| --- | --- |
| module `flashinfer.cake_lm_head_det_train` | the deterministic entries of `flashinfer.chunked_lm_head` (that module is removed; its non-deterministic entries are now `flashinfer.cake_lm_head_nondet_train`) |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_loss` | `flashinfer.chunked_lm_head.chunked_lm_head_det_loss`; before the `det` naming `chunked_lm_head_loss` |
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

### LM head + loss: non-deterministic package

The non-deterministic package `flashinfer.experimental.cake.cake_lm_head_nondet_train`
(module `flashinfer.cake_lm_head_nondet_train`)
keeps every precision boundary of the deterministic package (BF16 logits and
`dlogits`, FP32 loss arithmetic, FP32 `dX` and cross-chunk `dW` accumulation,
one final cast each) and relaxes only the run-to-run bitwise promise for the
gradients: the dX GEMM of a chunk of more than one K slice stores its partial
tiles into an FP32 slab, and a slab sum adds them in slice order and stores the
sum into the FP32 `dX` accumulator (a one-slice chunk stores its rows as the
deterministic package does). That fixed order repeats `dX` bit for bit, and its
bits may differ from the deterministic package's; even so, the package keeps
PyTorch's alert as part of its contract. `loss`, `logp` and `lse` keep
their fixed order. Its names carry `nondet` where the deterministic names carry `det`:

| Deterministic name | Non-deterministic name | The same name in the deterministic package |
| --- | --- | --- |
| package `flashinfer.experimental.cake.cake_lm_head_det_train` | package `flashinfer.experimental.cake.cake_lm_head_nondet_train` | none |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_loss` | `flashinfer.cake_lm_head_nondet_train.cake_lm_head_nondet_train_loss` | none |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_logprob` | `flashinfer.cake_lm_head_nondet_train.cake_lm_head_nondet_train_logprob` | none |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_preload` | `flashinfer.cake_lm_head_nondet_train.cake_lm_head_nondet_train_preload` | none |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_logprob` | `flashinfer.cake_lm_head_nondet_train.cake_lm_head_nondet_train_support_logprob` | none |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_kl` | `flashinfer.cake_lm_head_nondet_train.cake_lm_head_nondet_train_support_kl` | none |
| `cake_backend.cake_lm_head_det_train_{loss,logprob,logprob_temperature}` (also in `cake_backend_opt_in`), `cake_backend.cake_lm_head_det_train_logprob_keep` | `cake_backend.cake_lm_head_nondet_train_{loss,logprob,logprob_temperature}` (also in `cake_backend_opt_in`), `cake_backend.cake_lm_head_nondet_train_logprob_keep` | none |
| `cake_backend.prepare_lm_head_det_loss` | `cake_backend.prepare_lm_head_loss` | removed, no alias |
| `cake_backend.validate_lm_head_det_inputs` | `cake_backend.validate_lm_head_inputs` | removed, no alias |
| `cake_backend.lm_head_det_loss_workspace_size` | `cake_backend.lm_head_loss_workspace_size` | removed, no alias |
| `cake_backend.LmHeadDetLossRunner` | `cake_backend.LmHeadLossRunner` | removed, no alias |
| `cake_backend.CakeLMHeadDet{Loss,Logprob,LogprobTemperature}Function` (also in `cake_backend_opt_in`), `cake_backend.CakeLMHeadDetLogprobKeepFunction` | `cake_backend.CakeLMHeadNondet{Loss,Logprob,LogprobTemperature}Function` (also in `cake_backend_opt_in`), `cake_backend.CakeLMHeadNondetLogprobKeepFunction` | none |
| `cake_support.CakeLMHeadDetSupportFunction` | `cake_support.CakeLMHeadNondetSupportFunction` | none |
| `cake_jit.load_cake_lm_head_det_loss_module` | `cake_jit.load_cake_lm_head_loss_module` | removed, no alias |
| programs `cake_lm_head_det_loss_<20 hex>` | programs `cake_lm_head_loss_<20 hex>`, libraries `cake_lm_head_loss_<20 hex>_<arch>` | none |

The non-deterministic package defines these names in its own modules; the
deterministic modules define only the `det` names.

Mode and dispatch rules:

| Call | Result |
| --- | --- |
| `cake_lm_head_nondet_train_{loss,logprob,preload,support_logprob,support_kl}(...)` | the non-deterministic package, after the same argument checks as the deterministic entries and with the same `FLASHINFER_CAKE_LM_HEAD_LOSS_*` fallbacks; the module reads no torch flag and imports no deterministic module |
| the same while `torch.use_deterministic_algorithms(True)` is in effect (also `warn_only=True`) | the non-deterministic package; its alert decides (the last row); no fallback to the deterministic entry; a `k_slices` pin reaches it unchanged, and the module itself warns nothing |
| `cake_lm_head_det_train_{loss,logprob,preload,support_logprob,support_kl}(..., deterministic=<any>)`; likewise every function and autograd `Function` of the deterministic package | `TypeError` (none has a `deterministic` parameter); torch's deterministic mode changes nothing |
| `cake_lm_head_nondet_train_{loss,logprob,preload,support_logprob,support_kl}(..., deterministic=<any>)`; likewise every function and autograd `Function` of the non-deterministic package | `TypeError` (none has a `deterministic` parameter), raised before any work |
| a non-deterministic package entry while `torch.use_deterministic_algorithms(True)` is in effect | PyTorch's alert (`RuntimeError`, or a `UserWarning` with `warn_only=True`, after which the call runs) only right before a K-sliced dX launch: a `gemm_dx_red` chunk of more than one dX slice (the counted two-slice chunks of the zero-gradient row skip included) or the candidate-support backward's dense dX at more than one slice (`sm_100a` cells only); other launches do not alert (one-slice dX, the weight-gradient GEMMs in chunk order, the deferred last-chunk weight gradient, the budget recompute, the casts, the support sparse passes, forward-only calls) |

Zero-weight policy rows (`objective="policy"`, a valid label and
`loss_weights == 0`): the non-deterministic loss entry leaves them out of the
gradient path, as it does ignored rows.

| | `cake_lm_head_det_train_loss` | `cake_lm_head_nondet_train_loss` |
| --- | --- | --- |
| gradient rows of the chunk loop | every valid row | `(labels >= 0) & (loss_weights != 0)` (a NaN weight stays in) |
| `logp` of a zero-weight row (`return_logp=True`) | the loss path | a logp-only pass of the log-probability forward kernels |
| `loss`, `dX`, `dW` with finite inputs | the same mathematics | the same mathematics; `dW` and the loss sum group over other rows (bits may differ within the accuracy budgets) |
| a non-finite `X` row, logits or `infer_logp` in a zero-weight row | NaN through `0 * inf` | no effect on `loss` / `dX` / `dW` |
| host synchronizations | one count readback | one count readback (both counts in it) |
| not applied | -- | a call of fewer than `cake_backend.zero_weight_min_rows()` rows (default 4096), `compact_rows=False`, `objective="ce"`, the log-probability entry, the prepared runner, a caller-given `num_valid_rows`; `memory_mode="budget"` without `num_valid_rows` splits the rows (its bound still counts every valid row) |

Candidate-support entries (`cake_lm_head_nondet_train_support_logprob` /
`cake_lm_head_nondet_train_support_kl`):

| | `cake_lm_head_det_train_support_*` | `cake_lm_head_nondet_train_support_*` |
| --- | --- | --- |
| outputs, sparse `dX`, `dW` | the deterministic programs | the same programs: the same bits |
| dense `dX` | per vocabulary block an FP32 product block, added into the chunk sum in block order | each block's GEMM adds into the chunk sum from its epilogue: one K slice at every shape outside the measured cells of `support_dx_form` (the same bits at the same vocabulary block; the default block of this package, about 51712 columns, is wider than the deterministic package's, about 16384, so above about 16384 columns the blocks group differently and the bits differ); two or three K slices inside them (B200 only: five classes of H, token chunk and vocabulary block, at H 2048 to 6144): a slab sum of the slices in slice order, the same bits every run, within the accuracy budgets |
| global sparse `dW` zero fill | BF16 fill | FP32-word fill: the same bits |
| sparse `dW` long segments | parts summed in part order | the same |

The non-deterministic package adds no environment variable; the
`FLASHINFER_CAKE_LM_HEAD_LOSS_*` variables apply to the `det` and the `nondet`
entries, and its binding caches take the same two binding-cache settings.

## Family migration tables

Each table below takes effect when that family's rename PR merges; until then
the family's current names stay in effect.

Family rename PRs:

| Family | Mode | PR |
|---|---|---|
| LM head + loss | det | #8925 |
| LM head + loss | nondet | #8952 |
| DSA indexer | det | #8936 |
| GQA HD256 | nondet | #8939 |
| KDA | nondet (single rank and KCP) | KDA family PR |
| SwiGLU + MXFP8 quantization | det | #9100 |

### SwiGLU + MXFP8 quantization: deterministic package

| Old | New |
| --- | --- |
| module `flashinfer.gated_act_mxfp8` | `flashinfer.cake_gated_act_mxfp8_det_train` (stable API, `@flashinfer_api`; backend imported on first use) |
| package `flashinfer.cake_gated_act_mxfp8_backend` | `flashinfer.experimental.cake.cake_gated_act_mxfp8_det_train` |
| `silu_and_mul_mxfp8_quantize` | `cake_gated_act_mxfp8_det_train_forward` |
| `silu_and_mul_mxfp8_quantize_backward` | `cake_gated_act_mxfp8_det_train_backward` |
| `get_gated_act_mxfp8_module` | `cake_gated_act_mxfp8_det_train_register_custom_ops` |
| `GatedActMxfp8Outputs` | `CakeGatedActMXFP8DetOutputs` |
| backend `is_cake_gated_act_mxfp8_supported` | `is_cake_gated_act_mxfp8_det_train_supported` (also on the facade) |
| backend `check_inputs`, `empty_outputs`, `logical_outputs` | `cake_gated_act_mxfp8_det_train_check_inputs`, `_empty_outputs`, `_logical_outputs` |
| custom ops `flashinfer::silu_and_mul_mxfp8_quantize`, `flashinfer::silu_and_mul_mxfp8_quantize_backward` | `flashinfer::cake_gated_act_mxfp8_det_train_forward`, `flashinfer::cake_gated_act_mxfp8_det_train_backward` |

The JIT library names (`cake_gated_act_mxfp8_<hash>`) and the kernels are unchanged.

### GQA HD256: non-deterministic package

| Old | New |
| --- | --- |
| module `flashinfer.gqa_train` | `flashinfer.cake_gqa_nondet_train` (stable API, `@flashinfer_api`; backend imported on first use) |
| package `flashinfer.experimental.cake_gqa_train` | `flashinfer.experimental.cake.cake_gqa_nondet_train` |
| `varlen_gqa_attention` | `cake_gqa_nondet` |
| `varlen_gqa_attention_fwd` (backend `gqa_attention_fwd`) | `cake_gqa_nondet_train_forward` |
| `varlen_gqa_attention_bwd` (backend `gqa_attention_bwd`) | `cake_gqa_nondet_train_backward` |
| `gqa_train_workspace` | `cake_gqa_nondet_train_workspace` |
| `gqa_train_workspace_bytes` (backend `workspace_bytes`) | `cake_gqa_nondet_train_workspace_bytes` |
| backend `VarlenGQAAttentionFunction` | `CakeGQANondetFunction` |
| backend `Workspace` | `CakeGQANondetWorkspace` |
| backend `DkvAccSource` | `CakeGQANondetDkvAccSource` |

No aliases: the old names fail at import. No entry takes `deterministic=`. Every sum of the forward and the backward has one fixed order (the backward stores dS^T to a workspace, computes dQ as a GEMM over it and folds row-map destinations in window order; nothing adds with atomics), so every output repeats bit for bit run to run for one shape, `max_seqlen_k` and workspace budget, and nothing raises PyTorch's alert under `torch.use_deterministic_algorithms(True)`. `flashinfer.cake_gqa_det_train` is reserved (planned).

Build identities keep their names: program / library stems `cake_gqa_train_<20 hex>[_<arch>]`, kernel symbols `kernel_cake_gqa_train_<20 hex>`, csrc file names and `cake_jit.LIBRARY_PREFIX`; `build.json` changes only `"package"`. Internal backend helpers (`default_workspace`, `load`, `record`, `stage_record`, `library_name`, ...) keep their names.

### KDA: non-deterministic packages (single rank and KCP)

| Old | New |
| --- | --- |
| module `flashinfer.cake_kda_train` (mode selector, `deterministic=` keyword) | `flashinfer.cake_kda_nondet_train` (stable API, `@flashinfer_api`; backend imported on first use) |
| package `flashinfer.experimental.cake.cake_kda_train` | `flashinfer.experimental.cake.cake_kda_nondet_train` |
| `chunk_kda` | `cake_kda_nondet` |
| `chunk_kda_fwd` / `chunk_kda_bwd` | `cake_kda_nondet_train_forward` / `cake_kda_nondet_train_backward` |
| `kda_train_arena`, `kda_train_workspace`, `kda_train_workspace_bytes`, `kda_train_saved_bytes` | `cake_kda_nondet_train_arena`, `cake_kda_nondet_train_workspace`, `cake_kda_nondet_train_workspace_bytes`, `cake_kda_nondet_train_saved_bytes` |
| `is_kda_train_supported` | `is_cake_kda_nondet_train_supported` |
| backend `ChunkKDAFunction`, `KdaWorkspace`, `KdaChunkSaved`, `KdaArena` | `CakeKDANondetFunction`, `CakeKDANondetWorkspace`, `CakeKDANondetChunkSaved`, `CakeKDANondetArena` |
| module `flashinfer.cake_kda_train_kcp` | `flashinfer.cake_kda_nondet_train_kcp` (stable API, `@flashinfer_api`) |
| package `flashinfer.experimental.cake_kda_train_kcp` | `flashinfer.experimental.cake.cake_kda_nondet_train_kcp` |
| `chunk_kda_kcp` | `cake_kda_nondet_kcp` |
| `chunk_kda_fwd_kcp` / `chunk_kda_bwd_kcp` | `cake_kda_nondet_train_kcp_forward` / `cake_kda_nondet_train_kcp_backward` |
| `partition_kcp`, `build_context_kcp` | `cake_kda_nondet_train_kcp_partition`, `cake_kda_nondet_train_kcp_build_context` |
| `kda_workspace_bytes_kcp`, `kda_saved_bytes_kcp`, `kda_comm_bytes_kcp`, `kda_handoff_bytes_kcp` | `cake_kda_nondet_train_kcp_workspace_bytes`, `cake_kda_nondet_train_kcp_saved_bytes`, `cake_kda_nondet_train_kcp_comm_bytes`, `cake_kda_nondet_train_kcp_handoff_bytes` |
| `reduce_gate_param_grads_kcp`, `kcp_handoff_stats` | `cake_kda_nondet_train_kcp_reduce_gate_param_grads`, `cake_kda_nondet_train_kcp_handoff_stats` |
| `ContextKcp`, `PartitionKcp`, `WorkspaceKcp`, `SavedKcp`, `RecomputeHandleKcp`, `OptionsKcp` | `CakeKDANondetKcpContext`, `CakeKDANondetKcpPartition`, `CakeKDANondetKcpWorkspace`, `CakeKDANondetKcpSaved`, `CakeKDANondetKcpRecomputeHandle`, `CakeKDANondetKcpOptions` |
| backend `ChunkKDAFunctionKcp`, `CallKcp`, `FragmentKcp` | `CakeKDANondetKcpFunction`, `CakeKDANondetKcpCall`, `CakeKDANondetKcpFragment` |

No aliases: the old names fail at import. No entry takes `deterministic=`; the former selector default (`deterministic=True` ran the deterministic package) is gone, so a caller of the deterministic path imports `flashinfer.cake_kda_det_train` / `flashinfer.cake_kda_det_train_kcp`. No launch of these packages is unordered: under `torch.use_deterministic_algorithms(True)` (also with `warn_only=True`) the entries run the non-deterministic package and raise no alert and no warning; nothing falls back to the deterministic package. A workspace, saved set or arena option of the deterministic package is refused with `ValueError` naming the mode.

Build identities keep their names: program / library stems `cake_kda_train_<20 hex>[_kcp]`, kernel symbols, csrc file names and the `cake_jit` / `cake_kcp_jit` records; `build.json` changes only `"package"`, and the KCP package's `build.json` / `build.test.json` declare their single-rank sibling in `common.requires.packages` (`["cake_kda_nondet_train"]`). The KCP package imports that sibling by name from its own parent package (`_single_rank.py`), so vendoring builds ship the two directories side by side. Internal backend helpers (`plan`, `workspace_bytes`, `saved_bytes`, `kcp_plan`, `KcpPreconditionError`, ...) keep their names.
