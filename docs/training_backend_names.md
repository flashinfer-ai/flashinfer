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
| | nondet | single rank | `cake_mhc_head_nondet_train` | `cake_mhc_head_nondet` | `CakeMHCHeadNondetFunction` | planned |
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
| | nondet | single rank | `cake_dsa_indexer_nondet_train` | | | planned |
| DSA sparse attention | det | single rank | `cake_dsa_sparse_attention_det_train` | | | planned |
| | nondet | single rank | `cake_dsa_sparse_attention_nondet_train` | `cake_dsa_sparse_attention_nondet`, `cake_dsa_sparse_attention_nondet_varlen`, `cake_dsa_sparse_attention_nondet_nope`, `cake_dsa_sparse_attention_nondet_nope_varlen` | `CakeDSASparseAttentionNondetFunction` (both rope entries), `CakeDSASparseAttentionNondetNopeFunction`, `CakeDSASparseAttentionNondetNopeVarlenFunction` | `experimental.cake_dsa_train` / `dsa_sparse_attention.dsa_sparse_attention{,_varlen,_nope,_nope_varlen}` |
| GQA HD256 | det | single rank | `cake_gqa_det_train` | `cake_gqa_det` | `CakeGQADetFunction` | planned |
| | nondet | single rank | `cake_gqa_nondet_train` | `cake_gqa_nondet` | `CakeGQANondetFunction` | `gqa_train` / `varlen_gqa_attention` |

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
- **mHC head becomes its own family.** It ships `det` now; `nondet` is planned.
- **Missing versions keep their reserved names:** mHC head `nondet`, DSA indexer `nondet`, DSA sparse attention `det` and GQA `det`.

## Current names

| Training family | Canonical backend / route name |
| --- | --- |
| LM head + loss | `cake_lm_head_det_loss` (deterministic); `cake.cake_lm_head_loss` (non-deterministic) |

### LM head + loss

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

### LM head + loss: non-deterministic package

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
