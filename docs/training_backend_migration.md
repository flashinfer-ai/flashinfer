# CAKE training renames (#8928): migration from 0.7.3 and 0.7.4rc1

FlashInfer issue #8928 renames every CAKE training module to `flashinfer.cake_<op>_<det|nondet>_train`, with `_kcp` appended for a context-parallel (KCP) route. The module you import selects the mode: no entry point takes a `deterministic=` keyword, and no environment variable selects the mode. Each backend package moves as a whole directory, with its `build.json`, `cake_build.py` and csrc sources, to `flashinfer/experimental/cake/<module>/`. Old names and paths are removed without aliases, so importing an old name fails. Some old paths changed mode between releases, so look up your release (0.7.3 or 0.7.4rc1) and your old name or directory in the tables below; detailed naming rules are in `docs/training_backend_names.md`.

## Status and mode labels

- **PR** column: the PR that renames the family. Each family is renamed in its own PR, and all renames are released together, so you migrate once.
  - `#N (merged)`: the rename is in main.
  - `#N (open)`: the rename is in an open PR and takes effect when that PR merges.
  - `planned`: no PR exists yet. #8928 fixes the target name.
- Mode cells:
  - `det`: run-to-run bitwise deterministic forward and backward for fixed inputs, hardware and plan.
  - `nondet`: no run-to-run bitwise promise.
  - `nondet, det by default`: the package ran its non-deterministic programs only when called with `deterministic=False`. Its default call ran the deterministic programs.
  - `det alias of <dir>`: a directory without sources of its own that re-exported the deterministic package `<dir>`.
  - `-`: absent in that release.
- **In** column (Public names): the releases that shipped the old name. Names that existed only on main between releases are not listed.

## Target modules

Every shipped training module after the renames. Each module has a backend package at `flashinfer/experimental/cake/<module>/`.

| Family | Module | Mode | Route | PR |
| --- | --- | --- | --- | --- |
| RMSNorm | `flashinfer.cake_rmsnorm_det_train` | det | single rank | #8910 (open) |
|  | `flashinfer.cake_rmsnorm_nondet_train` | nondet | single rank | #8910 (open) |
| SiLU-gated RMSNorm | `flashinfer.cake_rmsnorm_silu_gate_det_train` | det | single rank | #8922 (merged) |
|  | `flashinfer.cake_rmsnorm_silu_gate_nondet_train` | nondet | single rank | #8922 (merged) |
| causal conv1d | `flashinfer.cake_causal_conv1d_det_train` | det | single rank | #8953 (open) |
|  | `flashinfer.cake_causal_conv1d_det_train_kcp` | det | KCP | #8953 (open) |
|  | `flashinfer.cake_causal_conv1d_nondet_train` | nondet | single rank | #8906 (merged) |
|  | `flashinfer.cake_causal_conv1d_nondet_train_kcp` | nondet | KCP | #8906 (merged) |
| mHC | `flashinfer.cake_mhc_det_train` | det | single rank | #8907 (open) |
|  | `flashinfer.cake_mhc_nondet_train` | nondet | single rank | #8907 (open) |
| mHC head | `flashinfer.cake_mhc_head_det_train` | det | single rank | #8907 (open) |
| GDN | `flashinfer.cake_gdn_det_train` | det | single rank | #8959 (open) |
|  | `flashinfer.cake_gdn_det_train_kcp` | det | KCP | #8959 (open) |
|  | `flashinfer.cake_gdn_nondet_train` | nondet | single rank | #8959 (open) |
|  | `flashinfer.cake_gdn_nondet_train_kcp` | nondet | KCP | #8959 (open) |
| KDA | `flashinfer.cake_kda_det_train` | det | single rank | planned |
|  | `flashinfer.cake_kda_det_train_kcp` | det | KCP | planned |
|  | `flashinfer.cake_kda_nondet_train` | nondet | single rank | planned |
|  | `flashinfer.cake_kda_nondet_train_kcp` | nondet | KCP | planned |
| MoE grouped GEMM | `flashinfer.cake_moe_grouped_gemm_det_train` | det | single rank | planned |
|  | `flashinfer.cake_moe_grouped_gemm_nondet_train` | nondet | single rank | planned |
| dense projection GEMM | `flashinfer.cake_dense_gemm_det_train` | det | single rank | #8926 (open) |
|  | `flashinfer.cake_dense_gemm_nondet_train` | nondet | single rank | #8926 (open) |
| LM head + loss | `flashinfer.cake_lm_head_det_train` | det | single rank | #8925 (open) |
|  | `flashinfer.cake_lm_head_nondet_train` | nondet | single rank | #8952 (open) |
| DSA indexer | `flashinfer.cake_dsa_indexer_det_train` | det | single rank | #8936 (merged) |
| DSA sparse attention | `flashinfer.cake_dsa_sparse_attention_nondet_train` | nondet | single rank | #8942 (open) |
| GQA HD256 | `flashinfer.cake_gqa_nondet_train` | nondet | single rank | #8939 (merged) |

## Calls that took `deterministic=`

| Old call (0.7.3 or 0.7.4rc1) | New call |
| --- | --- |
| an old entry called with the default, or with `deterministic=True` | the same entry of the family's `det` module |
| an old entry called with `deterministic=False` | the same entry of the family's `nondet` module |
| any new entry called with `deterministic=` | `TypeError`: no new entry has the parameter |
| an old module, old name or old backend path in the tables below | fails at import; there are no aliases |

## Old paths whose mode changed

| Old path | 0.7.3 | 0.7.4rc1 | Use now |
| --- | --- | --- | --- |
| `flashinfer/cake_rmsnorm_train_backend` (module `flashinfer.cake_rmsnorm_train`) | det | nondet, det by default (nondet since #8208) | 0.7.3 callers and 0.7.4rc1 default callers: `flashinfer.cake_rmsnorm_det_train`; 0.7.4rc1 `deterministic=False` callers: `flashinfer.cake_rmsnorm_nondet_train` |
| `flashinfer/cake_rmsnorm_silu_gate_train_backend` (module `flashinfer.cake_rmsnorm_silu_gate_train`) | det | nondet, det by default | 0.7.3 callers and 0.7.4rc1 default callers: `flashinfer.cake_rmsnorm_silu_gate_det_train`; 0.7.4rc1 `deterministic=False` callers: `flashinfer.cake_rmsnorm_silu_gate_nondet_train` |
| `flashinfer/experimental/dense_projection_gemm` | det | nondet, det by default | 0.7.3 callers and 0.7.4rc1 default callers: `cake_dense_gemm_det_train`; 0.7.4rc1 `deterministic=False` callers: `cake_dense_gemm_nondet_train` |

## Backend directories

For vendoring builds that copy a backend directory.

| Old directory | 0.7.3 | 0.7.4rc1 | New directory | PR | Note |
| --- | --- | --- | --- | --- | --- |
| `flashinfer/cake_rmsnorm_det_train_backend/` | - | det | `flashinfer/experimental/cake/cake_rmsnorm_det_train/` | #8910 (open) |  |
| `flashinfer/cake_rmsnorm_train_backend/` (as in 0.7.3) | det | nondet, det by default | `flashinfer/experimental/cake/cake_rmsnorm_det_train/` | #8910 (open) |  |
| `flashinfer/cake_rmsnorm_train_backend/` (as in 0.7.4rc1) | det | nondet, det by default | `flashinfer/experimental/cake/cake_rmsnorm_nondet_train/` | #8910 (open) |  |
| `flashinfer/cake_rmsnorm_silu_gate_det_train_backend/` | - | det | `flashinfer/experimental/cake/cake_rmsnorm_silu_gate_det_train/` | #8922 (merged) |  |
| `flashinfer/cake_rmsnorm_silu_gate_train_backend/` (as in 0.7.3) | det | nondet, det by default | `flashinfer/experimental/cake/cake_rmsnorm_silu_gate_det_train/` | #8922 (merged) |  |
| `flashinfer/cake_rmsnorm_silu_gate_train_backend/` (as in 0.7.4rc1) | det | nondet, det by default | `flashinfer/experimental/cake/cake_rmsnorm_silu_gate_nondet_train/` | #8922 (merged) |  |
| `flashinfer/experimental/cake/cake_causal_conv1d_train_backend` | - | nondet, det by default | `flashinfer/experimental/cake/cake_causal_conv1d_nondet_train` | #8906 (merged) |  |
| `flashinfer/cake_causal_conv1d_train_backend` | det | det alias | `flashinfer/experimental/cake/cake_causal_conv1d_det_train` | #8912 (merged), #8953 (open) |  |
| `flashinfer/cake_causal_conv1d_det_train_backend` | - | det | `flashinfer/experimental/cake/cake_causal_conv1d_det_train` | #8953 (open) |  |
| `flashinfer/cake_mhc_train_backend/` | det | det alias | `flashinfer/experimental/cake/cake_mhc_det_train/` | #8907 (open), #8900 (merged) |  |
| `flashinfer/cake_mhc_det_train_backend/` | - | det | `flashinfer/experimental/cake/cake_mhc_det_train/` | #8907 (open) |  |
| `flashinfer/experimental/cake/cake_mhc_train_backend/` | - | nondet, det by default | `flashinfer/experimental/cake/cake_mhc_nondet_train/` | #8907 (open) |  |
| `flashinfer/cake_mhc_train_head_backend/` | det | det alias | `flashinfer/experimental/cake/cake_mhc_head_det_train/` | #8907 (open), #8900 (merged) |  |
| `flashinfer/cake_mhc_det_train_head_backend/` | - | det | `flashinfer/experimental/cake/cake_mhc_head_det_train/` | #8907 (open) |  |
| `flashinfer/experimental/cake_gdn_det_train/` | det | det | `flashinfer/experimental/cake/cake_gdn_det_train/` | #8959 (open) |  |
| `flashinfer/experimental/cake_gdn_train/` | det | det alias | `flashinfer/experimental/cake/cake_gdn_det_train_kcp/` | #8921 (merged; removed the old path), #8959 (open) |  |
| `flashinfer/experimental/cake_gdn_det_train_kcp/` | - | det | `flashinfer/experimental/cake/cake_gdn_det_train_kcp/` | #8959 (open) |  |
| `flashinfer/experimental/cake/cake_gdn_train/` | - | nondet, det by default | `flashinfer/experimental/cake/cake_gdn_nondet_train/` | #8959 (open) |  |
| `flashinfer/experimental/cake_gdn_train_kcp/` | - | nondet, det by default | `flashinfer/experimental/cake/cake_gdn_nondet_train_kcp/` | #8959 (open) |  |
| `flashinfer/experimental/cake_kda_train/` (single-rank part) | det | det alias | `flashinfer/experimental/cake/cake_kda_det_train/` | planned |  |
| `flashinfer/experimental/cake_kda_det_train/` | - | det | `flashinfer/experimental/cake/cake_kda_det_train/` | planned |  |
| `flashinfer/experimental/cake_kda_train/` (KCP part) | det | det alias | `flashinfer/experimental/cake/cake_kda_det_train_kcp/` | planned | Vendor together with its single-rank sibling directory. |
| `flashinfer/experimental/cake_kda_det_train_kcp/` | - | det | `flashinfer/experimental/cake/cake_kda_det_train_kcp/` | planned | Vendor together with its single-rank sibling directory. |
| `flashinfer/experimental/cake/cake_kda_train/` | - | nondet, det by default | `flashinfer/experimental/cake/cake_kda_nondet_train/` | planned |  |
| `flashinfer/experimental/cake_kda_train_kcp/` | - | nondet, det by default | `flashinfer/experimental/cake/cake_kda_nondet_train_kcp/` | planned | Vendor together with its single-rank sibling directory. |
| `flashinfer/experimental/cake_moe_grouped_gemm/` | det | det alias | `flashinfer/experimental/cake/cake_moe_grouped_gemm_det_train/` | #8924 (merged; removed the old path), planned |  |
| `flashinfer/experimental/cake_moe_grouped_det_gemm/` | - | det | `flashinfer/experimental/cake/cake_moe_grouped_gemm_det_train/` | planned |  |
| `flashinfer/experimental/cake/cake_moe_grouped_gemm/` | - | nondet, det by default | `flashinfer/experimental/cake/cake_moe_grouped_gemm_nondet_train/` | planned |  |
| `flashinfer/experimental/dense_projection_det_gemm/` | - | det | `flashinfer/experimental/cake/cake_dense_gemm_det_train/` | #8926 (open) |  |
| `flashinfer/experimental/dense_projection_gemm/` (as in 0.7.3) | det | nondet, det by default | `flashinfer/experimental/cake/cake_dense_gemm_det_train/` | #8926 (open) |  |
| `flashinfer/experimental/dense_projection_gemm/` (as in 0.7.4rc1) | det | nondet, det by default | `flashinfer/experimental/cake/cake_dense_gemm_nondet_train/` | #8926 (open) | Needs its det sibling directory. |
| `flashinfer/experimental/cake_lm_head_loss` | det | det alias | `flashinfer/experimental/cake/cake_lm_head_det_train` | #8925 (open) |  |
| `flashinfer/experimental/cake_lm_head_det_loss` | - | det | `flashinfer/experimental/cake/cake_lm_head_det_train` | #8925 (open) |  |
| `flashinfer/experimental/cake/cake_lm_head_loss` | - | nondet, det by default | `flashinfer/experimental/cake/cake_lm_head_nondet_train` | #8952 (open) |  |
| `flashinfer/experimental/cake_dsa_indexer/` | det | det alias | `flashinfer/experimental/cake/cake_dsa_indexer_det_train/` | #8936 (merged) |  |
| `flashinfer/experimental/cake_dsa_indexer_det/` | - | det | `flashinfer/experimental/cake/cake_dsa_indexer_det_train/` | #8936 (merged) |  |
| `flashinfer/experimental/cake_dsa_train/` | nondet | nondet | `flashinfer/experimental/cake/cake_dsa_sparse_attention_nondet_train/` | #8942 (open) |  |
| `flashinfer/experimental/cake_gqa_train/` | nondet | nondet | `flashinfer/experimental/cake/cake_gqa_nondet_train/` | #8939 (merged) |  |

## Public names

### RMSNorm (cake_rmsnorm_det_train / cake_rmsnorm_nondet_train)

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.cake_rmsnorm_train (0.7.3; rc1 default deterministic=True) | 0.7.3, 0.7.4rc1 | flashinfer.cake_rmsnorm_det_train | #8910 (open) |
| module flashinfer.cake_rmsnorm_train with deterministic=False | 0.7.4rc1 | flashinfer.cake_rmsnorm_nondet_train | #8910 (open) |
| module flashinfer.cake_rmsnorm_det_train | 0.7.4rc1 | flashinfer.cake_rmsnorm_det_train (path unchanged; no-mode aliases and deterministic= removed; backend loaded on first use) | #8910 (open) |
| package flashinfer.cake_rmsnorm_det_train_backend | 0.7.4rc1 | flashinfer.experimental.cake.cake_rmsnorm_det_train | #8910 (open) |
| package flashinfer.cake_rmsnorm_train_backend (as in 0.7.3) | 0.7.3 | flashinfer.experimental.cake.cake_rmsnorm_det_train | #8910 (open) |
| package flashinfer.cake_rmsnorm_train_backend (as in 0.7.4rc1) | 0.7.4rc1 | flashinfer.experimental.cake.cake_rmsnorm_nondet_train | #8910 (open) |
| cake_rmsnorm (default / deterministic=True) | 0.7.3, 0.7.4rc1 | cake_rmsnorm_det | #8910 (open) |
| cake_rmsnorm(..., deterministic=False) | 0.7.4rc1 | cake_rmsnorm_nondet | #8910 (open) |
| CakeRMSNormFunction (default / deterministic=True) | 0.7.3, 0.7.4rc1 | CakeRMSNormDetFunction | #8910 (open) |
| CakeRMSNormFunction.apply(..., deterministic=False) | 0.7.4rc1 | CakeRMSNormNondetFunction | #8910 (open) |
| cake_rmsnorm_train_forward (default) | 0.7.3, 0.7.4rc1 | cake_rmsnorm_det_train_forward | #8910 (open) |
| cake_rmsnorm_train_forward(..., deterministic=False) | 0.7.4rc1 | cake_rmsnorm_nondet_train_forward | #8910 (open) |
| cake_rmsnorm_train_backward (default) | 0.7.3, 0.7.4rc1 | cake_rmsnorm_det_train_backward | #8910 (open) |
| cake_rmsnorm_train_backward(..., deterministic=False) | 0.7.4rc1 | cake_rmsnorm_nondet_train_backward | #8910 (open) |
| cake_rmsnorm_train_backward_workspace (default) | 0.7.3, 0.7.4rc1 | cake_rmsnorm_det_train_backward_workspace | #8910 (open) |
| cake_rmsnorm_train_backward_workspace(..., deterministic=False) | 0.7.4rc1 | cake_rmsnorm_nondet_train_backward_workspace | #8910 (open) |
| cake_rmsnorm_train_backward_workspace_bytes (default) | 0.7.3, 0.7.4rc1 | cake_rmsnorm_det_train_backward_workspace_bytes | #8910 (open) |
| cake_rmsnorm_train_backward_workspace_bytes(..., deterministic=False) | 0.7.4rc1 | cake_rmsnorm_nondet_train_backward_workspace_bytes | #8910 (open) |
| is_cake_rmsnorm_train_supported (default) | 0.7.3, 0.7.4rc1 | is_cake_rmsnorm_det_train_supported | #8910 (open) |
| is_cake_rmsnorm_train_supported(..., deterministic=False) | 0.7.4rc1 | is_cake_rmsnorm_nondet_train_supported | #8910 (open) |
| route_key on flashinfer.cake_rmsnorm_train / cake_rmsnorm_train_backend (0.7.3: det registry) | 0.7.3 | cake_rmsnorm_det_train_route_key | #8910 (open) |
| route_key on flashinfer.cake_rmsnorm_train / cake_rmsnorm_train_backend (0.7.4rc1: nondet registry) | 0.7.4rc1 | cake_rmsnorm_nondet_train_route_key | #8910 (open) |
| route_key on flashinfer.cake_rmsnorm_det_train / det backend | 0.7.4rc1 | cake_rmsnorm_det_train_route_key | #8910 (open) |
| program_windows (cake_rmsnorm_det_train / det backend) | 0.7.4rc1 | cake_rmsnorm_det_train_program_windows | #8910 (open) |
| weight_dtypes (cake_rmsnorm_det_train / det backend) | 0.7.4rc1 | cake_rmsnorm_det_train_weight_dtypes | #8910 (open) |
| flashinfer.cake_rmsnorm_train.{CakeRMSNormDetFunction, cake_rmsnorm_det, cake_rmsnorm_det_train_forward, cake_rmsnorm_det_train_backward, cake_rmsnorm_det_train_backward_workspace, cake_rmsnorm_det_train_backward_workspace_bytes, is_cake_rmsnorm_det_train_supported} (lazy __getattr__; same on cake_rmsnorm_train_backend) | 0.7.4rc1 | same names from flashinfer.cake_rmsnorm_det_train (or flashinfer.experimental.cake.cake_rmsnorm_det_train) | #8910 (open) |
| top-level flashinfer.{cake_rmsnorm, CakeRMSNormFunction, cake_rmsnorm_train_forward, cake_rmsnorm_train_backward, cake_rmsnorm_train_backward_workspace_bytes} | 0.7.3, 0.7.4rc1 | flashinfer.{cake_rmsnorm_det, CakeRMSNormDetFunction, cake_rmsnorm_det_train_forward, cake_rmsnorm_det_train_backward, cake_rmsnorm_det_train_backward_workspace_bytes} (lazy rows). No nondet name at top level: use flashinfer.cake_rmsnorm_nondet_train.<name> | #8910 (open) |
| top-level module attributes flashinfer.cake_rmsnorm_train, flashinfer.cake_rmsnorm_train_backend | 0.7.4rc1 | flashinfer.cake_rmsnorm_det_train / flashinfer.cake_rmsnorm_nondet_train (lazy module rows); backends are not bound at top level | #8910 (open) |
| keyword deterministic= (0.7.3: cake_rmsnorm, cake_rmsnorm_train_backward, CakeRMSNormFunction.forward, True only; rc1: every cake_rmsnorm_train entry, plus cake_rmsnorm_det / cake_rmsnorm_det_train_backward, True only) | 0.7.3, 0.7.4rc1 | removed; passing it raises TypeError | #8910 (open) |

### SiLU-gated RMSNorm (cake_rmsnorm_silu_gate_det_train / cake_rmsnorm_silu_gate_nondet_train)

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.cake_rmsnorm_silu_gate_train (0.7.3; rc1 default deterministic=True) | 0.7.3, 0.7.4rc1 | flashinfer.cake_rmsnorm_silu_gate_det_train | #8922 (merged) |
| module flashinfer.cake_rmsnorm_silu_gate_train with deterministic=False | 0.7.4rc1 | flashinfer.cake_rmsnorm_silu_gate_nondet_train | #8922 (merged) |
| module flashinfer.cake_rmsnorm_silu_gate_det_train | 0.7.4rc1 | flashinfer.cake_rmsnorm_silu_gate_det_train (path unchanged; no-mode aliases removed; backend loaded on first use) | #8922 (merged) |
| package flashinfer.cake_rmsnorm_silu_gate_det_train_backend | 0.7.4rc1 | flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train | #8922 (merged) |
| package flashinfer.cake_rmsnorm_silu_gate_train_backend (as in 0.7.3) | 0.7.3 | flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train | #8922 (merged) |
| package flashinfer.cake_rmsnorm_silu_gate_train_backend (as in 0.7.4rc1) | 0.7.4rc1 | flashinfer.experimental.cake.cake_rmsnorm_silu_gate_nondet_train | #8922 (merged) |
| cake_rmsnorm_silu_gate (default) | 0.7.3, 0.7.4rc1 | cake_rmsnorm_silu_gate_det | #8922 (merged) |
| cake_rmsnorm_silu_gate(..., deterministic=False) | 0.7.4rc1 | cake_rmsnorm_silu_gate_nondet | #8922 (merged) |
| CakeRMSNormSiLUGateFunction (default) | 0.7.3, 0.7.4rc1 | CakeRMSNormSiLUGateDetFunction | #8922 (merged) |
| CakeRMSNormSiLUGateFunction.apply(..., deterministic=False) | 0.7.4rc1 | CakeRMSNormSiLUGateNondetFunction | #8922 (merged) |
| cake_rmsnorm_silu_gate_train_forward (default) | 0.7.3, 0.7.4rc1 | cake_rmsnorm_silu_gate_det_train_forward | #8922 (merged) |
| cake_rmsnorm_silu_gate_train_forward(..., deterministic=False) | 0.7.4rc1 | cake_rmsnorm_silu_gate_nondet_train_forward | #8922 (merged) |
| cake_rmsnorm_silu_gate_train_backward (default) | 0.7.3, 0.7.4rc1 | cake_rmsnorm_silu_gate_det_train_backward | #8922 (merged) |
| cake_rmsnorm_silu_gate_train_backward(..., deterministic=False) | 0.7.4rc1 | cake_rmsnorm_silu_gate_nondet_train_backward | #8922 (merged) |
| cake_rmsnorm_silu_gate_train_backward_workspace_bytes (default) | 0.7.3, 0.7.4rc1 | cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes | #8922 (merged) |
| cake_rmsnorm_silu_gate_train_backward_workspace_bytes(..., deterministic=False) | 0.7.4rc1 | cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes | #8922 (merged) |
| cake_rmsnorm_silu_gate_train_backward_workspace (default) | 0.7.3, 0.7.4rc1 | cake_rmsnorm_silu_gate_det_train_backward_workspace | #8922 (merged) |
| cake_rmsnorm_silu_gate_train_backward_workspace(..., deterministic=False) | 0.7.4rc1 | cake_rmsnorm_silu_gate_nondet_train_backward_workspace | #8922 (merged) |
| cake_rmsnorm_silu_gate_train_reset_workspace | 0.7.4rc1 | cake_rmsnorm_silu_gate_nondet_train_reset_workspace (det: none; the det workspace needs no reset) | #8922 (merged) |
| is_cake_rmsnorm_silu_gate_train_supported (default) | 0.7.3, 0.7.4rc1 | is_cake_rmsnorm_silu_gate_det_train_supported | #8922 (merged) |
| is_cake_rmsnorm_silu_gate_train_supported(..., deterministic=False) | 0.7.4rc1 | is_cake_rmsnorm_silu_gate_nondet_train_supported | #8922 (merged) |
| top-level flashinfer.{cake_rmsnorm_silu_gate, CakeRMSNormSiLUGateFunction, cake_rmsnorm_silu_gate_train_forward, cake_rmsnorm_silu_gate_train_backward, cake_rmsnorm_silu_gate_train_backward_workspace_bytes} (default) | 0.7.3, 0.7.4rc1 | flashinfer.{cake_rmsnorm_silu_gate_det, CakeRMSNormSiLUGateDetFunction, cake_rmsnorm_silu_gate_det_train_forward, cake_rmsnorm_silu_gate_det_train_backward, cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes} (lazy rows) | #8922 (merged) |
| same top-level names called with deterministic=False | 0.7.4rc1 | flashinfer.{cake_rmsnorm_silu_gate_nondet, CakeRMSNormSiLUGateNondetFunction, cake_rmsnorm_silu_gate_nondet_train_forward, cake_rmsnorm_silu_gate_nondet_train_backward, cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes} (lazy rows) | #8922 (merged) |
| top-level module attributes flashinfer.cake_rmsnorm_silu_gate_train, flashinfer.cake_rmsnorm_silu_gate_train_backend, flashinfer.cake_rmsnorm_silu_gate_det_train_backend | 0.7.4rc1 | flashinfer.cake_rmsnorm_silu_gate_det_train / flashinfer.cake_rmsnorm_silu_gate_nondet_train (lazy module rows); backends are not bound at top level | #8922 (merged) |
| keyword deterministic= (rc1: every entry of the facade and the nondet backend; absent in 0.7.3) | 0.7.4rc1 | removed; passing it raises TypeError | #8922 (merged) |

### causal conv1d nondet (+KCP): cake_causal_conv1d_nondet_train[_kcp]

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.experimental.cake.cake_causal_conv1d_train | 0.7.4rc1 | flashinfer.cake_causal_conv1d_nondet_train | #8906 (merged) |
| module flashinfer.cake_causal_conv1d_train_kcp | 0.7.4rc1 | flashinfer.cake_causal_conv1d_nondet_train_kcp | #8906 (merged) |
| package flashinfer.experimental.cake.cake_causal_conv1d_train_backend (+ .cake_build, .cake_jit, .kcp, .kcp_jit) | 0.7.4rc1 | flashinfer.experimental.cake.cake_causal_conv1d_nondet_train (same submodules) | #8906 (merged) |
| cake_causal_conv1d (experimental.cake.cake_causal_conv1d_train + backend) | 0.7.4rc1 | cake_causal_conv1d_nondet | #8906 (merged) |
| CakeCausalConv1dFunction (experimental.cake.cake_causal_conv1d_train + backend) | 0.7.4rc1 | CakeCausalConv1dNondetFunction | #8906 (merged) |
| cake_causal_conv1d_train_forward / cake_causal_conv1d_train_backward (experimental.cake facade + backend) | 0.7.4rc1 | cake_causal_conv1d_nondet_train_forward / cake_causal_conv1d_nondet_train_backward | #8906 (merged) |
| cake_causal_conv1d_train_{forward,backward}_workspace{,_bytes} (experimental.cake facade + backend) | 0.7.4rc1 | cake_causal_conv1d_nondet_train_{forward,backward}_workspace{,_bytes} | #8906 (merged) |
| is_cake_causal_conv1d_train_supported (experimental.cake facade + backend) | 0.7.4rc1 | is_cake_causal_conv1d_nondet_train_supported | #8906 (merged) |
| cake_causal_conv1d_kcp (flashinfer.cake_causal_conv1d_train_kcp + backend .kcp) | 0.7.4rc1 | cake_causal_conv1d_nondet_kcp | #8906 (merged) |
| cake_causal_conv1d_multi_kcp (flashinfer.cake_causal_conv1d_train_kcp + backend .kcp) | 0.7.4rc1 | cake_causal_conv1d_nondet_kcp_multi | #8906 (merged) |
| RecomputeHandleKcp (flashinfer.cake_causal_conv1d_train_kcp + backend .kcp) | 0.7.4rc1 | CakeCausalConv1dNondetKcpRecomputeHandle | #8906 (merged) |
| backend .kcp CakeCausalConv1dFunctionKcp | 0.7.4rc1 | CakeCausalConv1dNondetKcpFunction | #8906 (merged) |
| backend .kcp CakeCausalConv1dMultiFunctionKcp | 0.7.4rc1 | CakeCausalConv1dNondetKcpMultiFunction | #8906 (merged) |
| backend .kcp cake_causal_conv1d_forward_kcp / cake_causal_conv1d_backward_kcp | 0.7.4rc1 | cake_causal_conv1d_nondet_train_kcp_forward / cake_causal_conv1d_nondet_train_kcp_backward | #8906 (merged) |
| backend .kcp cake_causal_conv1d_multi_forward_kcp / cake_causal_conv1d_multi_backward_kcp | 0.7.4rc1 | cake_causal_conv1d_nondet_train_kcp_multi_forward / cake_causal_conv1d_nondet_train_kcp_multi_backward | #8906 (merged) |
| backend .kcp graph_cache_clear / host_cache_clear | 0.7.4rc1 | cake_causal_conv1d_nondet_train_kcp_graph_cache_clear / cake_causal_conv1d_nondet_train_kcp_host_cache_clear | #8906 (merged) |
| deterministic= keyword (every rc1 nondet facade and backend entry, default True) | 0.7.4rc1 | removed (the module selects the mode); passing it raises TypeError | #8906 (merged) |

### causal conv1d det (+KCP): cake_causal_conv1d_det_train[_kcp]

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.cake_causal_conv1d_train | 0.7.3, 0.7.4rc1 | flashinfer.cake_causal_conv1d_det_train | #8912 (merged) |
| module flashinfer.cake_causal_conv1d_det_train | 0.7.4rc1 | flashinfer.cake_causal_conv1d_det_train (unchanged) | #8953 (open) |
| module flashinfer.cake_causal_conv1d_det_train_kcp | 0.7.4rc1 | flashinfer.cake_causal_conv1d_det_train_kcp (unchanged) | #8953 (open) |
| package flashinfer.cake_causal_conv1d_train_backend (+ .cake_build, .cake_jit, .kcp, .kcp_jit) | 0.7.3, 0.7.4rc1 | flashinfer.experimental.cake.cake_causal_conv1d_det_train (same submodules) | #8912 (merged), #8953 (open) |
| package flashinfer.cake_causal_conv1d_det_train_backend (+ .cake_build, .cake_jit, .kcp, .kcp_jit) | 0.7.4rc1 | flashinfer.experimental.cake.cake_causal_conv1d_det_train (same submodules) | #8953 (open) |
| flashinfer.cake_causal_conv1d (top level) | 0.7.3, 0.7.4rc1 | flashinfer.cake_causal_conv1d_det | #8912 (merged) |
| flashinfer.CakeCausalConv1dFunction (top level) | 0.7.3, 0.7.4rc1 | flashinfer.CakeCausalConv1dDetFunction | #8912 (merged) |
| flashinfer.cake_causal_conv1d_train_forward / flashinfer.cake_causal_conv1d_train_backward (top level) | 0.7.3, 0.7.4rc1 | flashinfer.cake_causal_conv1d_det_train_forward / flashinfer.cake_causal_conv1d_det_train_backward | #8912 (merged) |
| flashinfer.cake_causal_conv1d_train_forward_workspace_bytes / flashinfer.cake_causal_conv1d_train_backward_workspace_bytes (top level) | 0.7.3, 0.7.4rc1 | flashinfer.cake_causal_conv1d_det_train_forward_workspace_bytes / flashinfer.cake_causal_conv1d_det_train_backward_workspace_bytes | #8912 (merged) |
| cake_causal_conv1d (in flashinfer.cake_causal_conv1d_train and its backend; rc1 alias in det_train + det backend) | 0.7.3, 0.7.4rc1 | cake_causal_conv1d_det (flashinfer.cake_causal_conv1d_det_train) | #8912 (merged) |
| CakeCausalConv1dFunction (facade + backend; rc1 alias in det_train + det backend) | 0.7.3, 0.7.4rc1 | CakeCausalConv1dDetFunction | #8912 (merged) |
| cake_causal_conv1d_train_forward / cake_causal_conv1d_train_backward (facade + backend) | 0.7.3, 0.7.4rc1 | cake_causal_conv1d_det_train_forward / cake_causal_conv1d_det_train_backward | #8912 (merged) |
| cake_causal_conv1d_train_{forward,backward}_workspace{,_bytes} (facade + backend) | 0.7.3, 0.7.4rc1 | cake_causal_conv1d_det_train_{forward,backward}_workspace{,_bytes} | #8912 (merged) |
| is_cake_causal_conv1d_train_supported (facade + backend) | 0.7.3, 0.7.4rc1 | is_cake_causal_conv1d_det_train_supported | #8912 (merged) |
| cake_causal_conv1d_kcp (facade flashinfer.cake_causal_conv1d_train at 0.7.3 / flashinfer.cake_causal_conv1d_det_train alias at rc1; backend .kcp) | 0.7.3, 0.7.4rc1 | flashinfer.cake_causal_conv1d_det_train_kcp.cake_causal_conv1d_det_kcp (backend .kcp cake_causal_conv1d_det_kcp) | #8912 (merged) |
| cake_causal_conv1d_kcp_multi (rc1 alias on flashinfer.cake_causal_conv1d_det_train and det backend .kcp) | 0.7.4rc1 | flashinfer.cake_causal_conv1d_det_train_kcp.cake_causal_conv1d_det_kcp_multi | #8912 (merged), #8953 (open) |
| cake_causal_conv1d_det_multi_kcp (flashinfer.cake_causal_conv1d_det_train_kcp + det backend .kcp) | 0.7.4rc1 | cake_causal_conv1d_det_kcp_multi | #8953 (open) |
| KcpRecomputeHandle (facade flashinfer.cake_causal_conv1d_train at 0.7.3 / flashinfer.cake_causal_conv1d_det_train at rc1; backend .kcp) | 0.7.3, 0.7.4rc1 | flashinfer.cake_causal_conv1d_det_train_kcp.CakeCausalConv1dDetKcpRecomputeHandle | #8912 (merged), #8953 (open) |
| RecomputeHandleDetKcp (flashinfer.cake_causal_conv1d_det_train_kcp + det backend .kcp) | 0.7.4rc1 | CakeCausalConv1dDetKcpRecomputeHandle | #8953 (open) |
| backend .kcp CakeCausalConv1dKcpFunction | 0.7.3, 0.7.4rc1 | CakeCausalConv1dDetKcpFunction (also exported by flashinfer.cake_causal_conv1d_det_train_kcp) | #8912 (merged), #8953 (open) |
| backend .kcp CakeCausalConv1dDetFunctionKcp | 0.7.4rc1 | CakeCausalConv1dDetKcpFunction (also exported by flashinfer.cake_causal_conv1d_det_train_kcp) | #8953 (open) |
| backend .kcp CakeCausalConv1dDetMultiFunctionKcp (and its rc1 alias CakeCausalConv1dKcpMultiFunction) | 0.7.4rc1 | CakeCausalConv1dDetKcpMultiFunction (also exported by flashinfer.cake_causal_conv1d_det_train_kcp) | #8953 (open), #8912 (merged) |
| backend .kcp cake_causal_conv1d_kcp_forward / cake_causal_conv1d_kcp_backward | 0.7.3, 0.7.4rc1 | cake_causal_conv1d_det_train_kcp_forward / cake_causal_conv1d_det_train_kcp_backward | #8912 (merged), #8953 (open) |
| backend .kcp cake_causal_conv1d_det_forward_kcp / cake_causal_conv1d_det_backward_kcp | 0.7.4rc1 | cake_causal_conv1d_det_train_kcp_forward / cake_causal_conv1d_det_train_kcp_backward | #8953 (open) |
| backend .kcp cake_causal_conv1d_det_multi_forward_kcp / cake_causal_conv1d_det_multi_backward_kcp (and rc1 aliases cake_causal_conv1d_kcp_multi_forward / _backward) | 0.7.4rc1 | cake_causal_conv1d_det_train_kcp_multi_forward / cake_causal_conv1d_det_train_kcp_multi_backward | #8953 (open), #8912 (merged) |
| backend .kcp graph_cache_clear / host_cache_clear | 0.7.3, 0.7.4rc1 | cake_causal_conv1d_det_train_kcp_graph_cache_clear / cake_causal_conv1d_det_train_kcp_host_cache_clear | #8953 (open) |
| backend .kcp launch_grid | 0.7.4rc1 | _launch_grid (private, no public replacement) | #8953 (open) |
| error text "recompute_handle: the 'conv' entry belongs to another KcpContext" | 0.7.3, 0.7.4rc1 | "... belongs to another KCP context" | #8953 (open) |

### mHC (cake_mhc_det_train, cake_mhc_nondet_train)

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| flashinfer.cake_mhc_train (module; det at 0.7.3, det compatibility alias at 0.7.4rc1) | 0.7.3, 0.7.4rc1 | flashinfer.cake_mhc_det_train (head entries -> flashinfer.cake_mhc_head_det_train) | #8907 (open), #8900 (merged) |
| flashinfer.cake_mhc_det_train (module) | 0.7.4rc1 | flashinfer.cake_mhc_det_train (name unchanged; head entries split out; classes and route_key renamed; backend now under experimental/cake) | #8907 (open) |
| flashinfer.cake_mhc_train.cake_mhc_train_<verb> for verb in forward_prepare, forward_stats, forward_finish, backward_prepare, backward_finish, prepare, finish, forward_workspace_bytes, backward_workspace_bytes, forward_workspace, backward_workspace, reset_workspace; and the top-level flashinfer.cake_mhc_train_<verb> (top level has all of these except forward_workspace, backward_workspace and reset_workspace) | 0.7.3, 0.7.4rc1 | flashinfer.cake_mhc_det_train.cake_mhc_det_train_<verb>; the same 9 names are also at the top level as flashinfer.cake_mhc_det_train_<verb> | #8907 (open), #8900 (merged) |
| flashinfer.cake_mhc_det_train.cake_mhc_det_train_<verb> (same 12 verbs; 9 also at top level) | 0.7.4rc1 | unchanged: flashinfer.cake_mhc_det_train.cake_mhc_det_train_<verb> | #8907 (open) |
| flashinfer.cake_mhc_train.CakeMHCTrain{Prepare,Finish}Function and top-level flashinfer.CakeMHCTrain{Prepare,Finish}Function | 0.7.3, 0.7.4rc1 | flashinfer.cake_mhc_det_train.CakeMHCDet{Prepare,Finish}Function (also top-level flashinfer.CakeMHCDet{Prepare,Finish}Function through _LAZY_EXPORTS) | #8907 (open), #8900 (merged) |
| flashinfer.cake_mhc_det_train.CakeMHCDetTrain{Prepare,Finish}Function and top-level flashinfer.CakeMHCDetTrain{Prepare,Finish}Function | 0.7.4rc1 | flashinfer.cake_mhc_det_train.CakeMHCDet{Prepare,Finish}Function (also top level, lazy) | #8907 (open) |
| flashinfer.cake_mhc_train.is_cake_mhc_train_supported | 0.7.3, 0.7.4rc1 | flashinfer.cake_mhc_det_train.is_cake_mhc_det_train_supported | #8907 (open), #8900 (merged) |
| flashinfer.cake_mhc_train.route_key (0.7.3, 0.7.4rc1) and flashinfer.cake_mhc_det_train.route_key (0.7.4rc1) | 0.7.3, 0.7.4rc1 | flashinfer.cake_mhc_det_train.cake_mhc_det_train_route_key | #8907 (open) |
| flashinfer.cake_mhc_train_backend (0.7.3 det package; 0.7.4rc1 det alias) and flashinfer.cake_mhc_det_train_backend (0.7.4rc1) | 0.7.3, 0.7.4rc1 | flashinfer.experimental.cake.cake_mhc_det_train | #8907 (open), #8900 (merged) |
| flashinfer.experimental.cake.cake_mhc_train (module; selector, deterministic: bool = True) | 0.7.4rc1 | flashinfer.cake_mhc_nondet_train for deterministic=False callers; flashinfer.cake_mhc_det_train for default (deterministic=True) callers | #8907 (open) |
| flashinfer.experimental.cake.cake_mhc_train.cake_mhc_train_<verb>(..., deterministic=False) for the same 12 verbs (finish takes no keyword and follows the handle) | 0.7.4rc1 | flashinfer.cake_mhc_nondet_train.cake_mhc_nondet_train_<verb> (no deterministic keyword; not exported at top level) | #8907 (open) |
| flashinfer.experimental.cake.cake_mhc_train.cake_mhc_train_<verb>(...) with the default deterministic=True | 0.7.4rc1 | flashinfer.cake_mhc_det_train.cake_mhc_det_train_<verb> | #8907 (open) |
| flashinfer.experimental.cake.cake_mhc_train.is_cake_mhc_train_supported | 0.7.4rc1 | flashinfer.cake_mhc_nondet_train.is_cake_mhc_nondet_train_supported (deterministic=False); flashinfer.cake_mhc_det_train.is_cake_mhc_det_train_supported (default deterministic=True) | #8907 (open) |
| flashinfer.experimental.cake.cake_mhc_train.CakeMHCTrain{Prepare,Finish}Function (the nondet package's classes) | 0.7.4rc1 | flashinfer.cake_mhc_nondet_train.CakeMHCNondet{Prepare,Finish}Function | #8907 (open) |
| flashinfer.experimental.cake.cake_mhc_train.route_key | 0.7.4rc1 | flashinfer.cake_mhc_nondet_train.cake_mhc_nondet_train_route_key | #8907 (open) |
| flashinfer.experimental.cake.cake_mhc_train_backend (its own cake_mhc_train_* entries accept only deterministic=False) | 0.7.4rc1 | flashinfer.experimental.cake.cake_mhc_nondet_train (cake_mhc_nondet_train_* entries, no keyword) | #8907 (open) |

### mHC head (cake_mhc_head_det_train; nondet planned)

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| head part of flashinfer.cake_mhc_train (0.7.3; rc1 alias) and of flashinfer.cake_mhc_det_train (0.7.4rc1) | 0.7.3, 0.7.4rc1 | flashinfer.cake_mhc_head_det_train (new top-level facade module) | #8907 (open) |
| flashinfer.cake_mhc_train.cake_mhc_train_head_{forward,backward} and top-level flashinfer.cake_mhc_train_head_{forward,backward} (0.7.3, 0.7.4rc1); flashinfer.cake_mhc_det_train.cake_mhc_det_train_head_{forward,backward} and top-level flashinfer.cake_mhc_det_train_head_{forward,backward} (0.7.4rc1) | 0.7.3, 0.7.4rc1 | flashinfer.cake_mhc_head_det_train.cake_mhc_head_det_train_{forward,backward} (also top level) | #8907 (open), #8900 (merged) |
| flashinfer.cake_mhc_train.cake_mhc_train_head and top-level flashinfer.cake_mhc_train_head (0.7.3, 0.7.4rc1); flashinfer.cake_mhc_det_train.cake_mhc_det_train_head and top-level flashinfer.cake_mhc_det_train_head (0.7.4rc1) | 0.7.3, 0.7.4rc1 | flashinfer.cake_mhc_head_det_train.cake_mhc_head_det (also top level) | #8907 (open), #8900 (merged) |
| flashinfer.cake_mhc_train.CakeMHCTrainHeadFunction and top-level flashinfer.CakeMHCTrainHeadFunction (0.7.3, 0.7.4rc1); flashinfer.cake_mhc_det_train.CakeMHCDetTrainHeadFunction and top-level flashinfer.CakeMHCDetTrainHeadFunction (0.7.4rc1) | 0.7.3, 0.7.4rc1 | flashinfer.cake_mhc_head_det_train.CakeMHCHeadDetFunction (also top level, lazy) | #8907 (open), #8900 (merged) |
| flashinfer.cake_mhc_train.is_cake_mhc_train_head_supported (0.7.3, 0.7.4rc1); flashinfer.cake_mhc_det_train.is_cake_mhc_det_train_head_supported (0.7.4rc1) | 0.7.3, 0.7.4rc1 | flashinfer.cake_mhc_head_det_train.is_cake_mhc_head_det_train_supported | #8907 (open) |
| flashinfer.cake_mhc_train_head_backend (0.7.3 det package; 0.7.4rc1 det alias) and flashinfer.cake_mhc_det_train_head_backend (0.7.4rc1) | 0.7.3, 0.7.4rc1 | flashinfer.experimental.cake.cake_mhc_head_det_train | #8907 (open), #8900 (merged) |
| (none) | none | flashinfer.cake_mhc_head_nondet_train / cake_mhc_head_nondet / CakeMHCHeadNondetFunction (reserved) | planned |

### GDN det, single rank (cake_gdn_det_train)

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.experimental.cake_gdn_det_train (backend package imported directly; no top-level facade existed) | 0.7.3, 0.7.4rc1 | flashinfer.cake_gdn_det_train (new top-level facade); backend flashinfer.experimental.cake.cake_gdn_det_train | #8959 (open) |
| chunk_gated_delta_rule_det (flashinfer.experimental.cake_gdn_det_train) | 0.7.4rc1 | flashinfer.cake_gdn_det_train.cake_gdn_det | #8959 (open) |
| chunk_gated_delta_rule (flashinfer.experimental.cake_gdn_det_train; det entry at 0.7.3, det alias at 0.7.4rc1) | 0.7.3, 0.7.4rc1 | flashinfer.cake_gdn_det_train.cake_gdn_det (alias dropped) | #8959 (open) |
| ChunkGatedDeltaRuleDetFunction | 0.7.4rc1 | CakeGDNDetFunction | #8959 (open) |
| ChunkGatedDeltaRuleFunction (det package; class at 0.7.3, alias at 0.7.4rc1) | 0.7.3, 0.7.4rc1 | CakeGDNDetFunction (alias dropped) | #8959 (open) |
| gdn_chunk_det_forward / gdn_chunk_det_backward | 0.7.4rc1 | cake_gdn_det_train_forward / cake_gdn_det_train_backward | #8959 (open) |
| gdn_chunk_forward / gdn_chunk_backward (det package; functions at 0.7.3, aliases at 0.7.4rc1) | 0.7.3, 0.7.4rc1 | cake_gdn_det_train_forward / cake_gdn_det_train_backward (aliases dropped) | #8959 (open) |

### GDN det KCP (cake_gdn_det_train_kcp)

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.gdn_train (det KCP facade at 0.7.3; legacy alias module at 0.7.4rc1) | 0.7.3, 0.7.4rc1 | flashinfer.cake_gdn_det_train_kcp | #8959 (open) |
| backend package flashinfer.experimental.cake_gdn_train (det KCP at 0.7.3; alias at 0.7.4rc1) | 0.7.3, 0.7.4rc1 | flashinfer.experimental.cake.cake_gdn_det_train_kcp | #8959 (open) |
| module flashinfer.cake_gdn_det_train_kcp | 0.7.4rc1 | flashinfer.cake_gdn_det_train_kcp (name unchanged; backend moves) | #8959 (open) |
| backend package flashinfer.experimental.cake_gdn_det_train_kcp | 0.7.4rc1 | flashinfer.experimental.cake.cake_gdn_det_train_kcp | #8959 (open) |
| chunk_gdn_det_kcp (flashinfer.cake_gdn_det_train_kcp) | 0.7.4rc1 | cake_gdn_det_kcp | #8959 (open) |
| chunk_gdn_kcp (flashinfer.gdn_train / experimental.cake_gdn_train; det; alias at 0.7.4rc1) | 0.7.3, 0.7.4rc1 | flashinfer.cake_gdn_det_train_kcp.cake_gdn_det_kcp | #8959 (open) |
| ChunkGDNDetFunctionKcp; ChunkGDNKcpFunction (0.7.3 class, 0.7.4rc1 alias) | 0.7.3 (ChunkGDNKcpFunction), 0.7.4rc1 (both) | CakeGDNDetKcpFunction | #8959 (open) |
| chunk_gdn_det_fwd_kcp / chunk_gdn_det_bwd_kcp; chunk_gdn_kcp_fwd / chunk_gdn_kcp_bwd (flashinfer.gdn_train, 0.7.4rc1 alias) | 0.7.3 (chunk_gdn_kcp_fwd/bwd), 0.7.4rc1 (both) | cake_gdn_det_train_kcp_forward / cake_gdn_det_train_kcp_backward | #8959 (open) |
| partition_det_kcp; kcp_partition (flashinfer.gdn_train, 0.7.4rc1 alias) | 0.7.3 (kcp_partition), 0.7.4rc1 (both) | cake_gdn_det_train_kcp_partition | #8959 (open) |
| build_context_det_kcp; build_kcp_context (flashinfer.gdn_train, 0.7.4rc1 alias) | 0.7.3 (build_kcp_context), 0.7.4rc1 (both) | cake_gdn_det_train_kcp_build_context | #8959 (open) |
| gdn_det_workspace_bytes_kcp / gdn_det_saved_bytes_kcp / gdn_det_comm_bytes_kcp; gdn_kcp_workspace_bytes / gdn_kcp_saved_bytes / gdn_kcp_comm_bytes (flashinfer.gdn_train, 0.7.4rc1 alias) | 0.7.3 (gdn_kcp_*), 0.7.4rc1 (both) | cake_gdn_det_train_kcp_workspace_bytes / cake_gdn_det_train_kcp_saved_bytes / cake_gdn_det_train_kcp_comm_bytes | #8959 (open) |
| is_gdn_det_train_kcp_supported; is_gdn_train_supported (flashinfer.gdn_train, 0.7.4rc1 alias) | 0.7.3 (is_gdn_train_supported), 0.7.4rc1 (both) | is_cake_gdn_det_train_kcp_supported | #8959 (open) |
| ContextDetKcp, OptionsDetKcp, PartitionDetKcp, RecomputeHandleDetKcp, SavedDetKcp, WorkspaceDetKcp | 0.7.4rc1 | CakeGDNDetKcpContext, CakeGDNDetKcpOptions, CakeGDNDetKcpPartition, CakeGDNDetKcpRecomputeHandle, CakeGDNDetKcpSaved, CakeGDNDetKcpWorkspace | #8959 (open) |
| KcpContext, KcpOptions, KcpPartition, KcpRecomputeHandle, KcpSaved, KcpWorkspace (flashinfer.gdn_train / experimental.cake_gdn_train; aliases at 0.7.4rc1) | 0.7.3, 0.7.4rc1 | CakeGDNDetKcpContext, CakeGDNDetKcpOptions, CakeGDNDetKcpPartition, CakeGDNDetKcpRecomputeHandle, CakeGDNDetKcpSaved, CakeGDNDetKcpWorkspace | #8959 (open) |
| packed_qkv_views_det_kcp | 0.7.4rc1 | cake_gdn_det_train_kcp_packed_qkv_views | #8959 (open) |

### GDN nondet, single rank (cake_gdn_nondet_train)

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.experimental.cake.cake_gdn_train | 0.7.4rc1 | flashinfer.cake_gdn_nondet_train (new top-level facade); backend flashinfer.experimental.cake.cake_gdn_nondet_train | #8959 (open) |
| chunk_gated_delta_rule(..., deterministic=False) (flashinfer.experimental.cake.cake_gdn_train) | 0.7.4rc1 | flashinfer.cake_gdn_nondet_train.cake_gdn_nondet(...) (no deterministic keyword) | #8959 (open) |
| ChunkGatedDeltaRuleFunction (flashinfer.experimental.cake.cake_gdn_train) | 0.7.4rc1 | CakeGDNNondetFunction | #8959 (open) |
| gdn_chunk_forward / gdn_chunk_backward (..., deterministic=False) (flashinfer.experimental.cake.cake_gdn_train) | 0.7.4rc1 | cake_gdn_nondet_train_forward / cake_gdn_nondet_train_backward | #8959 (open) |

### GDN nondet KCP (cake_gdn_nondet_train_kcp)

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.cake_gdn_train_kcp | 0.7.4rc1 | flashinfer.cake_gdn_nondet_train_kcp | #8959 (open) |
| a flashinfer.cake_gdn_train_kcp entry called with the default deterministic (or deterministic=True) | 0.7.4rc1 | the matching det KCP entry: flashinfer.cake_gdn_det_train_kcp.cake_gdn_det_kcp, cake_gdn_det_train_kcp_forward/_backward/_partition/_build_context/_*_bytes, is_cake_gdn_det_train_kcp_supported | #8959 (open) |
| backend package flashinfer.experimental.cake_gdn_train_kcp | 0.7.4rc1 | flashinfer.experimental.cake.cake_gdn_nondet_train_kcp | #8959 (open) |
| chunk_gdn_kcp(..., deterministic=False) (flashinfer.cake_gdn_train_kcp) | 0.7.4rc1 | cake_gdn_nondet_kcp(...) (no deterministic keyword) | #8959 (open) |
| ChunkGDNFunctionKcp (flashinfer.experimental.cake_gdn_train_kcp) | 0.7.4rc1 | CakeGDNNondetKcpFunction | #8959 (open) |
| chunk_gdn_fwd_kcp / chunk_gdn_bwd_kcp (..., deterministic=False) | 0.7.4rc1 | cake_gdn_nondet_train_kcp_forward / cake_gdn_nondet_train_kcp_backward | #8959 (open) |
| partition_kcp | 0.7.4rc1 | cake_gdn_nondet_train_kcp_partition | #8959 (open) |
| build_context_kcp | 0.7.4rc1 | cake_gdn_nondet_train_kcp_build_context | #8959 (open) |
| gdn_workspace_bytes_kcp / gdn_saved_bytes_kcp / gdn_comm_bytes_kcp | 0.7.4rc1 | cake_gdn_nondet_train_kcp_workspace_bytes / cake_gdn_nondet_train_kcp_saved_bytes / cake_gdn_nondet_train_kcp_comm_bytes | #8959 (open) |
| is_gdn_train_kcp_supported | 0.7.4rc1 | is_cake_gdn_nondet_train_kcp_supported | #8959 (open) |
| ContextKcp, OptionsKcp, PartitionKcp, RecomputeHandleKcp, SavedKcp, WorkspaceKcp | 0.7.4rc1 | CakeGDNNondetKcpContext, CakeGDNNondetKcpOptions, CakeGDNNondetKcpPartition, CakeGDNNondetKcpRecomputeHandle, CakeGDNNondetKcpSaved, CakeGDNNondetKcpWorkspace | #8959 (open) |
| packed_qkv_views_kcp (flashinfer.experimental.cake_gdn_train_kcp) | 0.7.4rc1 | cake_gdn_nondet_train_kcp_packed_qkv_views | #8959 (open) |

### KDA det (single rank): cake_kda_det_train

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.kda_train (single-rank entries) | 0.7.3, 0.7.4rc1 | flashinfer.cake_kda_det_train | #8908 (open) |
| flashinfer.kda_train.chunk_kda_fwd | 0.7.3, 0.7.4rc1 | flashinfer.cake_kda_det_train.cake_kda_det_train_forward | planned |
| flashinfer.kda_train.chunk_kda_bwd | 0.7.3, 0.7.4rc1 | flashinfer.cake_kda_det_train.cake_kda_det_train_backward | planned |
| flashinfer.kda_train.chunk_kda | 0.7.3, 0.7.4rc1 | flashinfer.cake_kda_det_train.cake_kda_det | planned |
| flashinfer.kda_train.kda_train_arena | 0.7.3, 0.7.4rc1 | flashinfer.cake_kda_det_train.cake_kda_det_train_arena | planned |
| flashinfer.kda_train.kda_train_workspace | 0.7.3, 0.7.4rc1 | flashinfer.cake_kda_det_train.cake_kda_det_train_workspace | planned |
| flashinfer.kda_train.kda_train_workspace_bytes | 0.7.3, 0.7.4rc1 | flashinfer.cake_kda_det_train.cake_kda_det_train_workspace_bytes | planned |
| flashinfer.kda_train.kda_train_saved_bytes | 0.7.3, 0.7.4rc1 | flashinfer.cake_kda_det_train.cake_kda_det_train_saved_bytes | planned |
| flashinfer.kda_train.is_kda_train_supported | 0.7.3, 0.7.4rc1 | flashinfer.cake_kda_det_train.is_cake_kda_det_train_supported | planned |
| module flashinfer.cake_kda_det_train | 0.7.4rc1 | flashinfer.cake_kda_det_train (unchanged) | planned |
| flashinfer.cake_kda_det_train.chunk_kda_det_fwd | 0.7.4rc1 | cake_kda_det_train_forward | planned |
| flashinfer.cake_kda_det_train.chunk_kda_det_bwd | 0.7.4rc1 | cake_kda_det_train_backward | planned |
| flashinfer.cake_kda_det_train.chunk_kda_det | 0.7.4rc1 | cake_kda_det | planned |
| flashinfer.cake_kda_det_train.kda_det_train_arena | 0.7.4rc1 | cake_kda_det_train_arena | planned |
| flashinfer.cake_kda_det_train.kda_det_train_workspace | 0.7.4rc1 | cake_kda_det_train_workspace | planned |
| flashinfer.cake_kda_det_train.kda_det_train_workspace_bytes | 0.7.4rc1 | cake_kda_det_train_workspace_bytes | planned |
| flashinfer.cake_kda_det_train.kda_det_train_saved_bytes | 0.7.4rc1 | cake_kda_det_train_saved_bytes | planned |
| flashinfer.cake_kda_det_train.is_kda_det_train_supported | 0.7.4rc1 | is_cake_kda_det_train_supported | planned |
| package flashinfer.experimental.cake_kda_train (and submodules cake_backend, cake_jit, cake_build) | 0.7.3, 0.7.4rc1 | flashinfer.experimental.cake.cake_kda_det_train (same submodule names) | planned |
| package flashinfer.experimental.cake_kda_det_train | 0.7.4rc1 | flashinfer.experimental.cake.cake_kda_det_train | planned |
| ChunkKDAFunction (experimental.cake_kda_train[.cake_backend]; rc1 alias also in experimental.cake_kda_det_train[.cake_backend]) | 0.7.3, 0.7.4rc1 | CakeKDADetFunction | planned |
| ChunkKDADetFunction | 0.7.4rc1 | CakeKDADetFunction | planned |
| KdaChunkSaved | 0.7.3, 0.7.4rc1 | CakeKDADetChunkSaved | planned |
| KdaDetChunkSaved | 0.7.4rc1 | CakeKDADetChunkSaved | planned |
| KdaWorkspace | 0.7.3, 0.7.4rc1 | CakeKDADetWorkspace | planned |
| KdaDetWorkspace | 0.7.4rc1 | CakeKDADetWorkspace | planned |
| KdaArena (cake_backend) | 0.7.3, 0.7.4rc1 | CakeKDADetArena | planned |
| KdaDetArena (cake_backend) | 0.7.4rc1 | CakeKDADetArena | planned |
| backend chunk_kda / chunk_kda_fwd / chunk_kda_bwd (experimental.cake_kda_train; rc1 aliases in experimental.cake_kda_det_train[.cake_backend]) | 0.7.3, 0.7.4rc1 | cake_kda_det / cake_kda_det_train_forward / cake_kda_det_train_backward | planned |
| backend chunk_kda_det / chunk_kda_det_fwd / chunk_kda_det_bwd | 0.7.4rc1 | cake_kda_det / cake_kda_det_train_forward / cake_kda_det_train_backward | planned |
| tests.test_helpers.cake_kda_train_reference | 0.7.3, 0.7.4rc1 | tests.test_helpers.cake_kda_det_train_reference | #8908 (open) |

### KDA det KCP: cake_kda_det_train_kcp

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.kda_train (KCP entries) | 0.7.3, 0.7.4rc1 | flashinfer.cake_kda_det_train_kcp | #8908 (open) |
| kcp_partition (flashinfer.kda_train; experimental.cake_kda_train.cake_kcp_plan; rc1 alias in det cake_kcp_plan) | 0.7.3, 0.7.4rc1 | cake_kda_det_train_kcp_partition | planned |
| build_kcp_context | 0.7.3, 0.7.4rc1 | cake_kda_det_train_kcp_build_context | planned |
| chunk_kda_kcp_fwd | 0.7.3, 0.7.4rc1 | cake_kda_det_train_kcp_forward | planned |
| chunk_kda_kcp_bwd | 0.7.3, 0.7.4rc1 | cake_kda_det_train_kcp_backward | planned |
| chunk_kda_kcp (flashinfer.kda_train / det backend) | 0.7.3, 0.7.4rc1 | cake_kda_det_kcp | planned |
| kcp_workspace_bytes | 0.7.3, 0.7.4rc1 | cake_kda_det_train_kcp_workspace_bytes | planned |
| kcp_saved_bytes | 0.7.3, 0.7.4rc1 | cake_kda_det_train_kcp_saved_bytes | planned |
| kcp_comm_bytes | 0.7.3, 0.7.4rc1 | cake_kda_det_train_kcp_comm_bytes | planned |
| kcp_reduce_gate_param_grads (flashinfer.kda_train only) | 0.7.4rc1 | cake_kda_det_train_kcp_reduce_gate_param_grads | planned |
| KcpRecomputeHandle | 0.7.3, 0.7.4rc1 | CakeKDADetKcpRecomputeHandle | planned |
| KcpContext | 0.7.3, 0.7.4rc1 | CakeKDADetKcpContext | planned |
| KcpPartition | 0.7.3, 0.7.4rc1 | CakeKDADetKcpPartition | planned |
| KcpFragment | 0.7.3, 0.7.4rc1 | CakeKDADetKcpFragment | planned |
| KcpWorkspace | 0.7.3, 0.7.4rc1 | CakeKDADetKcpWorkspace | planned |
| KcpSaved | 0.7.3, 0.7.4rc1 | CakeKDADetKcpSaved | planned |
| KcpOptions | 0.7.3, 0.7.4rc1 | CakeKDADetKcpOptions | planned |
| KcpCall | 0.7.3, 0.7.4rc1 | CakeKDADetKcpCall | planned |
| ChunkKDAKcpFunction | 0.7.3, 0.7.4rc1 | CakeKDADetKcpFunction | planned |
| module flashinfer.cake_kda_det_train_kcp | 0.7.4rc1 | flashinfer.cake_kda_det_train_kcp (unchanged) | planned |
| partition_det_kcp | 0.7.4rc1 | cake_kda_det_train_kcp_partition | planned |
| build_context_det_kcp | 0.7.4rc1 | cake_kda_det_train_kcp_build_context | planned |
| chunk_kda_det_fwd_kcp | 0.7.4rc1 | cake_kda_det_train_kcp_forward | planned |
| chunk_kda_det_bwd_kcp | 0.7.4rc1 | cake_kda_det_train_kcp_backward | planned |
| chunk_kda_det_kcp | 0.7.4rc1 | cake_kda_det_kcp | planned |
| kda_det_workspace_bytes_kcp | 0.7.4rc1 | cake_kda_det_train_kcp_workspace_bytes | planned |
| kda_det_saved_bytes_kcp | 0.7.4rc1 | cake_kda_det_train_kcp_saved_bytes | planned |
| kda_det_comm_bytes_kcp | 0.7.4rc1 | cake_kda_det_train_kcp_comm_bytes | planned |
| reduce_gate_param_grads_det_kcp | 0.7.4rc1 | cake_kda_det_train_kcp_reduce_gate_param_grads | planned |
| ContextDetKcp | 0.7.4rc1 | CakeKDADetKcpContext | planned |
| PartitionDetKcp | 0.7.4rc1 | CakeKDADetKcpPartition | planned |
| WorkspaceDetKcp | 0.7.4rc1 | CakeKDADetKcpWorkspace | planned |
| SavedDetKcp | 0.7.4rc1 | CakeKDADetKcpSaved | planned |
| RecomputeHandleDetKcp | 0.7.4rc1 | CakeKDADetKcpRecomputeHandle | planned |
| OptionsDetKcp | 0.7.4rc1 | CakeKDADetKcpOptions | planned |
| FragmentDetKcp (package only) | 0.7.4rc1 | CakeKDADetKcpFragment | planned |
| CallDetKcp (package only) | 0.7.4rc1 | CakeKDADetKcpCall | planned |
| ChunkKDADetFunctionKcp (package only) | 0.7.4rc1 | CakeKDADetKcpFunction | planned |
| package flashinfer.experimental.cake_kda_det_train_kcp (submodules cake_kcp, cake_kcp_plan, cake_kcp_jit, cake_build) | 0.7.4rc1 | flashinfer.experimental.cake.cake_kda_det_train_kcp (same submodules) | planned |
| modules flashinfer.experimental.cake_kda_train.cake_kcp / .cake_kcp_plan / .cake_kcp_jit | 0.7.3, 0.7.4rc1 | flashinfer.experimental.cake.cake_kda_det_train_kcp.cake_kcp / .cake_kcp_plan / .cake_kcp_jit | planned |

### KDA nondet (single rank): cake_kda_nondet_train

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.cake_kda_train | 0.7.4rc1 | flashinfer.cake_kda_nondet_train (for deterministic=False callers); default deterministic=True callers go to flashinfer.cake_kda_det_train | planned |
| flashinfer.cake_kda_train.chunk_kda_fwd | 0.7.4rc1 | flashinfer.cake_kda_nondet_train.cake_kda_nondet_train_forward | planned |
| flashinfer.cake_kda_train.chunk_kda_bwd | 0.7.4rc1 | flashinfer.cake_kda_nondet_train.cake_kda_nondet_train_backward | planned |
| flashinfer.cake_kda_train.chunk_kda | 0.7.4rc1 | flashinfer.cake_kda_nondet_train.cake_kda_nondet | planned |
| flashinfer.cake_kda_train.kda_train_arena | 0.7.4rc1 | flashinfer.cake_kda_nondet_train.cake_kda_nondet_train_arena | planned |
| flashinfer.cake_kda_train.kda_train_workspace | 0.7.4rc1 | flashinfer.cake_kda_nondet_train.cake_kda_nondet_train_workspace | planned |
| flashinfer.cake_kda_train.kda_train_workspace_bytes | 0.7.4rc1 | flashinfer.cake_kda_nondet_train.cake_kda_nondet_train_workspace_bytes | planned |
| flashinfer.cake_kda_train.kda_train_saved_bytes | 0.7.4rc1 | flashinfer.cake_kda_nondet_train.cake_kda_nondet_train_saved_bytes | planned |
| flashinfer.cake_kda_train.is_kda_train_supported | 0.7.4rc1 | flashinfer.cake_kda_nondet_train.is_cake_kda_nondet_train_supported | planned |
| keyword deterministic= on every flashinfer.cake_kda_train entry (default True) | 0.7.4rc1 | removed: True (default) -> the matching flashinfer.cake_kda_det_train entry; False -> the cake_kda_nondet_train entry without the keyword | planned |
| package flashinfer.experimental.cake.cake_kda_train (submodules cake_backend, cake_jit, cake_build) | 0.7.4rc1 | flashinfer.experimental.cake.cake_kda_nondet_train (same submodules) | planned |
| ChunkKDAFunction (experimental.cake.cake_kda_train) | 0.7.4rc1 | CakeKDANondetFunction | planned |
| KdaWorkspace (experimental.cake.cake_kda_train) | 0.7.4rc1 | CakeKDANondetWorkspace | planned |
| KdaChunkSaved (experimental.cake.cake_kda_train) | 0.7.4rc1 | CakeKDANondetChunkSaved | planned |
| KdaArena (experimental.cake.cake_kda_train.cake_backend) | 0.7.4rc1 | CakeKDANondetArena | planned |
| backend chunk_kda / chunk_kda_fwd / chunk_kda_bwd (experimental.cake.cake_kda_train) | 0.7.4rc1 | cake_kda_nondet / cake_kda_nondet_train_forward / cake_kda_nondet_train_backward | planned |

### KDA nondet KCP: cake_kda_nondet_train_kcp

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.cake_kda_train_kcp | 0.7.4rc1 | flashinfer.cake_kda_nondet_train_kcp (for deterministic=False callers); default deterministic=True callers go to flashinfer.cake_kda_det_train_kcp | planned |
| flashinfer.cake_kda_train_kcp.chunk_kda_kcp | 0.7.4rc1 | cake_kda_nondet_kcp | planned |
| chunk_kda_fwd_kcp | 0.7.4rc1 | cake_kda_nondet_train_kcp_forward | planned |
| chunk_kda_bwd_kcp | 0.7.4rc1 | cake_kda_nondet_train_kcp_backward | planned |
| partition_kcp | 0.7.4rc1 | cake_kda_nondet_train_kcp_partition | planned |
| build_context_kcp | 0.7.4rc1 | cake_kda_nondet_train_kcp_build_context | planned |
| kda_workspace_bytes_kcp | 0.7.4rc1 | cake_kda_nondet_train_kcp_workspace_bytes | planned |
| kda_saved_bytes_kcp | 0.7.4rc1 | cake_kda_nondet_train_kcp_saved_bytes | planned |
| kda_comm_bytes_kcp | 0.7.4rc1 | cake_kda_nondet_train_kcp_comm_bytes | planned |
| reduce_gate_param_grads_kcp | 0.7.4rc1 | cake_kda_nondet_train_kcp_reduce_gate_param_grads | planned |
| kda_handoff_bytes_kcp (nondet only) | 0.7.4rc1 | cake_kda_nondet_train_kcp_handoff_bytes | planned |
| kcp_handoff_stats (nondet only) | 0.7.4rc1 | cake_kda_nondet_train_kcp_handoff_stats | planned |
| ContextKcp | 0.7.4rc1 | CakeKDANondetKcpContext | planned |
| PartitionKcp | 0.7.4rc1 | CakeKDANondetKcpPartition | planned |
| WorkspaceKcp | 0.7.4rc1 | CakeKDANondetKcpWorkspace | planned |
| SavedKcp | 0.7.4rc1 | CakeKDANondetKcpSaved | planned |
| RecomputeHandleKcp | 0.7.4rc1 | CakeKDANondetKcpRecomputeHandle | planned |
| OptionsKcp | 0.7.4rc1 | CakeKDANondetKcpOptions | planned |
| ChunkKDAFunctionKcp (package only) | 0.7.4rc1 | CakeKDANondetKcpFunction | planned |
| CallKcp (package only) | 0.7.4rc1 | CakeKDANondetKcpCall | planned |
| FragmentKcp (package only) | 0.7.4rc1 | CakeKDANondetKcpFragment | planned |
| keyword deterministic= on every flashinfer.cake_kda_train_kcp entry except kcp_handoff_stats (default True) | 0.7.4rc1 | removed: True (default) -> the matching flashinfer.cake_kda_det_train_kcp entry; False -> the cake_kda_nondet_train_kcp entry without the keyword | planned |
| package flashinfer.experimental.cake_kda_train_kcp (submodules cake_kcp, cake_kcp_plan, cake_kcp_jit, cake_build) | 0.7.4rc1 | flashinfer.experimental.cake.cake_kda_nondet_train_kcp (same submodules) | planned |

### MoE grouped GEMM (det + nondet): cake_moe_grouped_gemm_det_train / cake_moe_grouped_gemm_nondet_train

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| package flashinfer.experimental.cake_moe_grouped_gemm (one level) + submodules cake_autograd, cake_backend, cake_build, cake_jit | 0.7.3, 0.7.4rc1 | removed, no alias (#8924); det target module flashinfer.cake_moe_grouped_gemm_det_train (planned) | planned |
| package flashinfer.experimental.cake_moe_grouped_det_gemm | 0.7.4rc1 | module flashinfer.cake_moe_grouped_gemm_det_train (backend package flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train) | planned |
| package flashinfer.experimental.cake.cake_moe_grouped_gemm | 0.7.4rc1 | module flashinfer.cake_moe_grouped_gemm_nondet_train (backend package flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train) | planned |
| facade module flashinfer.experimental.cake.grouped_mm (cake_grouped_mm, grouped_gemm_{fwd,dgrad,wgrad}) | 0.7.4rc1 | no #8928 target stated. Callers that passed deterministic=False move to flashinfer.cake_moe_grouped_gemm_nondet_train. Callers that used the default (deterministic=True) move to flashinfer.cake_moe_grouped_gemm_det_train. | planned |
| module flashinfer.grouped_mm.cake | 0.7.3, 0.7.4rc1 | interim flashinfer.grouped_mm.cake_det (#8924, main only); #8928 final target not stated (det family module: flashinfer.cake_moe_grouped_gemm_det_train) | planned |
| cake_grouped_det_mm (det package root and cake_autograd, flashinfer.grouped_mm, flashinfer.grouped_mm.cake) | 0.7.4rc1 | cake_moe_grouped_gemm_det | planned |
| cake_grouped_mm, det side (flashinfer.grouped_mm, flashinfer.grouped_mm.cake, experimental.cake_moe_grouped_gemm[.cake_autograd]) | 0.7.3 (0.7.4rc1: deprecated det alias) | cake_moe_grouped_gemm_det | planned |
| CakeGroupedDetMm | 0.7.4rc1 | CakeMoEGroupedGemmDetFunction | planned |
| CakeGroupedMm, det side (experimental.cake_moe_grouped_gemm[.cake_autograd]) | 0.7.3 (0.7.4rc1: deprecated det alias) | CakeMoEGroupedGemmDetFunction | planned |
| cake_grouped_mm, nondet side (experimental.cake.cake_moe_grouped_gemm[.cake_autograd]; facade experimental.cake.grouped_mm.cake_grouped_mm(deterministic=False)) | 0.7.4rc1 | cake_moe_grouped_gemm_nondet | planned |
| CakeGroupedMm, nondet side (experimental.cake.cake_moe_grouped_gemm[.cake_autograd]) | 0.7.4rc1 | CakeMoEGroupedGemmNondetFunction | planned |
| grouped_det_gemm_{fwd,dgrad,wgrad} (det package, flashinfer.grouped_mm, flashinfer.grouped_mm.cake) | 0.7.4rc1 | cake_moe_grouped_gemm_det_train_* (checklist; exact verb spelling not fixed; rule 4 requires forward, not fwd) | planned |
| grouped_gemm_{fwd,dgrad,wgrad}, det side (flashinfer.grouped_mm, flashinfer.grouped_mm.cake, experimental.cake_moe_grouped_gemm) | 0.7.3 (0.7.4rc1: deprecated det aliases) | cake_moe_grouped_gemm_det_train_* (exact verbs not fixed) | planned |
| prepare_grouped_gemm_{fwd,dgrad,wgrad}, GroupedGemmLaunch (0.7.3 det) / prepare_grouped_det_gemm_{fwd,dgrad,wgrad}, GroupedDetGemmLaunch (rc1 det) | 0.7.3, 0.7.4rc1 | not stated in #8928 (by rules 4/6: cake_moe_grouped_gemm_det_train_<verb>, CakeMoEGroupedGemmDet<Noun>) | planned |
| nondet grouped_gemm_{fwd,dgrad,wgrad}, prepare_grouped_gemm_{fwd,dgrad,wgrad}, GroupedGemmLaunch (experimental.cake.cake_moe_grouped_gemm; facade grouped_gemm_* with deterministic=False) | 0.7.4rc1 | not stated in #8928 (by rules 4/6: cake_moe_grouped_gemm_nondet_train_<verb>, CakeMoEGroupedGemmNondet<Noun>) | planned |
| keyword deterministic= (0.7.3: cake_grouped_mm(deterministic=True), and False ran the same det programs; rc1: det entries the same, nondet entries required False, facade selected the mode) | 0.7.3, 0.7.4rc1 | removed; the module selects the mode | planned |

### dense projection GEMM (det + nondet): cake_dense_gemm_det_train / cake_dense_gemm_nondet_train

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| package flashinfer.experimental.dense_projection_det_gemm (cake_backend, cake_jit, cake_build) | 0.7.4rc1 | module flashinfer.cake_dense_gemm_det_train (backend package flashinfer.experimental.cake.cake_dense_gemm_det_train) | #8926 (open) |
| package flashinfer.experimental.dense_projection_gemm, as used at 0.7.3 or through the rc1 default deterministic=True | 0.7.3, 0.7.4rc1 | module flashinfer.cake_dense_gemm_det_train | #8926 (open) |
| package flashinfer.experimental.dense_projection_gemm called with deterministic=False | 0.7.4rc1 | module flashinfer.cake_dense_gemm_nondet_train (backend package flashinfer.experimental.cake.cake_dense_gemm_nondet_train) | #8926 (open) |
| dense_projection_det_gemm (rc1 det cake_backend); dense_projection_gemm (0.7.3 det; rc1 det alias; rc1 nondet package default) | 0.7.3, 0.7.4rc1 | cake_dense_gemm_det_train_projection | #8926 (open) |
| dense_projection_gemm(..., deterministic=False) (rc1 dense_projection_gemm.cake_backend) | 0.7.4rc1 | cake_dense_gemm_nondet_train_projection | #8926 (open) |
| projection_det_{forward,dgrad,wgrad} (rc1); projection_{forward,dgrad,wgrad} (0.7.3 det; rc1 det aliases / nondet default) | 0.7.3, 0.7.4rc1 | cake_dense_gemm_det_train_projection_{forward,dgrad,wgrad} | #8926 (open) |
| projection_{forward,dgrad,wgrad}(..., deterministic=False) (rc1) | 0.7.4rc1 | cake_dense_gemm_nondet_train_projection_{forward,dgrad,wgrad} | #8926 (open) |
| plan_dense_projection_det_gemm / prepare_dense_projection_det_gemm / prepare_projection_det_wgrad (rc1); plan_dense_projection_gemm / prepare_dense_projection_gemm / prepare_projection_wgrad (0.7.3 det; rc1 det aliases) | 0.7.3, 0.7.4rc1 | cake_dense_gemm_det_train_projection_plan / _projection_prepare / _projection_wgrad_prepare | #8926 (open) |
| plan_dense_projection_gemm / prepare_dense_projection_gemm / prepare_projection_wgrad (..., deterministic=False, **kwargs) (rc1) | 0.7.4rc1 | cake_dense_gemm_nondet_train_projection_plan / _projection_prepare / _projection_wgrad_prepare (no **kwargs; explicit keyword-only knobs = det planner knobs + splits, chains) | #8926 (open) |
| router_fp32_det_gemm (rc1); router_fp32_gemm (0.7.3 det; rc1 det alias) | 0.7.3, 0.7.4rc1 | cake_dense_gemm_det_train_router | #8926 (open) |
| router_fp32_gemm(..., deterministic=False) (rc1) | 0.7.4rc1 | cake_dense_gemm_nondet_train_router | #8926 (open) |
| router_det_{forward,dgrad,wgrad} (rc1); router_{forward,dgrad,wgrad} (0.7.3 det; rc1 det aliases) | 0.7.3, 0.7.4rc1 | cake_dense_gemm_det_train_router_{forward,dgrad,wgrad} | #8926 (open) |
| router_{forward,dgrad,wgrad}(..., deterministic=False) (rc1) | 0.7.4rc1 | cake_dense_gemm_nondet_train_router_{forward,dgrad,wgrad} | #8926 (open) |
| plan_router_fp32_det_gemm / prepare_router_fp32_det_gemm (rc1); plan_router_fp32_gemm / prepare_router_fp32_gemm (0.7.3 det; rc1 det aliases) | 0.7.3, 0.7.4rc1 | cake_dense_gemm_det_train_router_plan / _router_prepare | #8926 (open) |
| plan_router_fp32_gemm / prepare_router_fp32_gemm (..., deterministic=False) (rc1) | 0.7.4rc1 | cake_dense_gemm_nondet_train_router_plan / _router_prepare | #8926 (open) |
| clear_convenience_caches (0.7.3 det; rc1 in both packages) | 0.7.3, 0.7.4rc1 | cake_dense_gemm_det_train_convenience_cache_clear / cake_dense_gemm_nondet_train_convenience_cache_clear | #8926 (open) |
| DetGemmPlan / PreparedDetGemm / RouterDetPlan / PreparedRouterDetGemm (rc1); GemmPlan / PreparedGemm / RouterPlan / PreparedRouterGemm (0.7.3 det; rc1 det aliases and nondet read-through) | 0.7.3, 0.7.4rc1 | CakeDenseGemmDetProjectionPlan / CakeDenseGemmDetProjectionLaunch / CakeDenseGemmDetRouterPlan / CakeDenseGemmDetRouterLaunch | #8926 (open) |
| AtomicGemmPlan / PreparedAtomicGemm / AtomicRouterPlan / PreparedAtomicRouterGemm (rc1) | 0.7.4rc1 | CakeDenseGemmNondetProjectionPlan / CakeDenseGemmNondetProjectionLaunch / CakeDenseGemmNondetRouterPlan / CakeDenseGemmNondetRouterLaunch | #8926 (open) |
| (none) | new (no old name) | is_cake_dense_gemm_det_train_supported(device) / is_cake_dense_gemm_nondet_train_supported(device) | #8926 (open) |
| keyword deterministic= (rc1 dense_projection_gemm.cake_backend entries, default True) | 0.7.4rc1 | removed (TypeError); the module selects the mode | #8926 (open) |
| det former aliases (GemmPlan, PreparedGemm, PreparedRouterGemm, RouterPlan, the 13 mode-less functions, split_fp32_to_bf16x3_) in dense_projection_det_gemm.cake_backend; det cake_jit.load_cake_dense_projection_gemm_module; nondet cake_backend read-through (__getattr__/_DET_TYPES), DETERMINISTIC_NAMES, nondet instance_symbol / split_fp32_to_bf16x3_ | 0.7.4rc1 (the mode-less spellings were the det originals at 0.7.3; see rows above) | removed, no alias (det cake_backend.instance_symbol and nondet cake_jit.load_cake_dense_projection_gemm_module stay as backend internals) | #8926 (open) |

### LM head det (cake_lm_head_det_train)

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.chunked_lm_head (det entries / default-mode calls) | 0.7.3, 0.7.4rc1 | module flashinfer.cake_lm_head_det_train (stable API, @flashinfer_api; also lazy top-level attribute flashinfer.cake_lm_head_det_train) | #8925 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_det_loss | 0.7.4rc1 | flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_loss | #8925 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_det_logprob | 0.7.4rc1 | flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_logprob | #8925 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_det_preload | 0.7.4rc1 | flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_preload | #8925 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_det_support_logprob | 0.7.4rc1 | flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_logprob | #8925 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_det_support_kl | 0.7.4rc1 | flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_kl | #8925 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_loss(...) with default / deterministic=True | 0.7.3, 0.7.4rc1 | flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_loss(...) | #8925 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_logprob(...) with default / deterministic=True | 0.7.3, 0.7.4rc1 | flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_logprob(...) | #8925 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_preload(...) with default / deterministic=True | 0.7.3, 0.7.4rc1 | flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_preload(...) | #8925 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_support_logprob(...) / chunked_lm_head_support_kl(...) with default / deterministic=True | 0.7.4rc1 | flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_logprob(...) / cake_lm_head_det_train_support_kl(...) | #8925 (open) |
| chunked_lm_head_*(...) without deterministic=False (det call form) | 0.7.3, 0.7.4rc1 | ValueError naming the det entry (interim, #8925 only); module removed by #8952 | #8925 (open) |
| package flashinfer.experimental.cake_lm_head_det_loss (submodules cake_backend, cake_backend_opt_in, cake_support, cake_jit, cake_build) | 0.7.4rc1 | package flashinfer.experimental.cake.cake_lm_head_det_train (same submodules) | #8925 (open) |
| package flashinfer.experimental.cake_lm_head_loss (one level; det package at 0.7.3, det alias package at 0.7.4rc1) | 0.7.3, 0.7.4rc1 | package flashinfer.experimental.cake.cake_lm_head_det_train (alias removed, no replacement alias) | #8925 (open) |
| cake_lm_head_det_loss.cake_backend.chunked_lm_head_det_{loss,logprob,logprob_temperature,logprob_keep}; cake_backend_opt_in.chunked_lm_head_det_{loss,logprob,logprob_temperature} | 0.7.4rc1 | cake_lm_head_det_train.cake_backend.cake_lm_head_det_train_{loss,logprob,logprob_temperature,logprob_keep}; cake_backend_opt_in.cake_lm_head_det_train_{loss,logprob,logprob_temperature} | #8925 (open) |
| flashinfer.experimental.cake_lm_head_loss.cake_backend.chunked_lm_head_{loss,logprob} and cake_backend_opt_in.chunked_lm_head_{loss,logprob} (0.7.3 det names; at 0.7.4rc1 pre-det aliases in cake_lm_head_det_loss and via the alias package) | 0.7.3, 0.7.4rc1 | cake_lm_head_det_train.cake_backend.cake_lm_head_det_train_{loss,logprob} (same in cake_backend_opt_in) | #8925 (open) |
| cake_backend_opt_in.preload (flashinfer.experimental.cake_lm_head_loss at 0.7.3; flashinfer.experimental.cake_lm_head_det_loss at 0.7.4rc1) | 0.7.3, 0.7.4rc1 | cake_lm_head_det_train.cake_backend_opt_in.cake_lm_head_det_train_preload | #8925 (open) |
| cake_lm_head_det_loss.cake_support.chunked_lm_head_det_support_{logprob,kl} | 0.7.4rc1 | cake_lm_head_det_train.cake_support.cake_lm_head_det_train_support_{logprob,kl} | #8925 (open) |
| cake_lm_head_det_loss.cake_support.chunked_lm_head_support_{logprob,kl} (pre-det aliases; also via flashinfer.experimental.cake_lm_head_loss.cake_support) | 0.7.4rc1 | cake_lm_head_det_train.cake_support.cake_lm_head_det_train_support_{logprob,kl} | #8925 (open) |
| ChunkedLmHeadDet{Loss,Logprob,LogprobTemperature}Function (cake_backend and cake_backend_opt_in of cake_lm_head_det_loss) | 0.7.4rc1 | CakeLMHeadDet{Loss,Logprob,LogprobTemperature}Function (same modules of cake_lm_head_det_train) | #8925 (open) |
| cake_backend.ChunkedLmHeadDetLogprobKeepFunction | 0.7.4rc1 | cake_backend.CakeLMHeadDetLogprobKeepFunction | #8925 (open) |
| ChunkedLmHead{Loss,Logprob}Function in cake_backend / cake_backend_opt_in (0.7.3 det classes of flashinfer.experimental.cake_lm_head_loss; 0.7.4rc1 pre-det aliases in cake_lm_head_det_loss) | 0.7.3, 0.7.4rc1 | CakeLMHeadDet{Loss,Logprob}Function | #8925 (open) |
| cake_support.SupportDetLogprobFunction | 0.7.4rc1 | cake_support.CakeLMHeadDetSupportFunction (serves both support entries) | #8925 (open) |
| cake_lm_head_det_loss.cake_support.SupportLogprobFunction (pre-det alias) | 0.7.4rc1 | cake_support.CakeLMHeadDetSupportFunction | #8925 (open) |
| pre-det internal names validate_lm_head_inputs, lm_head_loss_workspace_size, LmHeadLossRunner, prepare_lm_head_loss (cake_backend, cake_backend_opt_in), load_cake_lm_head_loss_module (cake_jit, cake_backend, cake_backend_opt_in) | 0.7.3, 0.7.4rc1 | removed, no alias; internal det names kept (not forwarded): validate_lm_head_det_inputs, lm_head_det_loss_workspace_size, LmHeadDetLossRunner, prepare_lm_head_det_loss, cake_jit.load_cake_lm_head_det_loss_module in flashinfer.experimental.cake.cake_lm_head_det_train | #8925 (open) |
| deterministic= keyword on det entries, backend entries and autograd Function.forward (only True accepted) | 0.7.3, 0.7.4rc1 | removed; passing it raises TypeError; torch.use_deterministic_algorithms has no effect | #8925 (open) |

### LM head nondet (cake_lm_head_nondet_train)

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| module flashinfer.chunked_lm_head (its chunked_lm_head_* dispatchers, deterministic=False path) | 0.7.4rc1 (module also in 0.7.3, but it had no nondet path there) | module flashinfer.cake_lm_head_nondet_train (stable API, @flashinfer_api; also lazy top-level attribute flashinfer.cake_lm_head_nondet_train) | #8952 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_loss(..., deterministic=False) | 0.7.4rc1 | flashinfer.cake_lm_head_nondet_train.cake_lm_head_nondet_train_loss(...) | #8952 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_logprob(..., deterministic=False) | 0.7.4rc1 | flashinfer.cake_lm_head_nondet_train.cake_lm_head_nondet_train_logprob(...) | #8952 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_preload(..., deterministic=False) | 0.7.4rc1 | flashinfer.cake_lm_head_nondet_train.cake_lm_head_nondet_train_preload(...) | #8952 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_support_logprob(..., deterministic=False) | 0.7.4rc1 | flashinfer.cake_lm_head_nondet_train.cake_lm_head_nondet_train_support_logprob(...) | #8952 (open) |
| flashinfer.chunked_lm_head.chunked_lm_head_support_kl(..., deterministic=False) | 0.7.4rc1 | flashinfer.cake_lm_head_nondet_train.cake_lm_head_nondet_train_support_kl(...) | #8952 (open) |
| package flashinfer.experimental.cake.cake_lm_head_loss (cake_backend, cake_backend_opt_in, cake_support, cake_jit, cake_build) | 0.7.4rc1 | package flashinfer.experimental.cake.cake_lm_head_nondet_train (same submodules) | #8952 (open) |
| cake.cake_lm_head_loss.cake_backend.chunked_lm_head_{loss,logprob,logprob_temperature,logprob_keep}; cake_backend_opt_in.chunked_lm_head_{loss,logprob,logprob_temperature} | 0.7.4rc1 | cake_lm_head_nondet_train.cake_backend.cake_lm_head_nondet_train_{loss,logprob,logprob_temperature,logprob_keep}; cake_backend_opt_in.cake_lm_head_nondet_train_{loss,logprob,logprob_temperature} | #8952 (open) |
| cake.cake_lm_head_loss.cake_backend_opt_in.preload | 0.7.4rc1 | cake_lm_head_nondet_train.cake_backend_opt_in.cake_lm_head_nondet_train_preload | #8952 (open) |
| cake.cake_lm_head_loss.cake_support.chunked_lm_head_support_{logprob,kl} | 0.7.4rc1 | cake_lm_head_nondet_train.cake_support.cake_lm_head_nondet_train_support_{logprob,kl} | #8952 (open) |
| ChunkedLmHead{Loss,Logprob,LogprobTemperature}Function (cake_backend and cake_backend_opt_in of cake.cake_lm_head_loss) | 0.7.4rc1 | CakeLMHeadNondet{Loss,Logprob,LogprobTemperature}Function (both backend modules) | #8952 (open) |
| cake.cake_lm_head_loss.cake_backend.ChunkedLmHeadLogprobKeepFunction | 0.7.4rc1 | cake_lm_head_nondet_train.cake_backend.CakeLMHeadNondetLogprobKeepFunction | #8952 (open) |
| cake.cake_lm_head_loss.cake_support.SupportLogprobFunction | 0.7.4rc1 | cake_lm_head_nondet_train.cake_support.CakeLMHeadNondetSupportFunction (serves both support entries) | #8952 (open) |
| cake_support.__all__ also listing SupportReferenceEngine, SupportRows, decode_support_error, describe_support_error, support_error_word, support_memory_report, support_program_available, validate_support_inputs | 0.7.4rc1 | __all__ = the two support entries, CakeLMHeadNondetSupportFunction and the contract constants; the helpers stay module attributes | #8952 (open) |
| deterministic keyword of every ND entry (facade dispatchers and backend entries; backend default True raised ValueError) and the deterministic position of every autograd Function | 0.7.4rc1 | removed; passing it raises Python's TypeError | #8952 (open) |

### DSA indexer det (cake_dsa_indexer_det_train)

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| flashinfer.dsa_indexer (Cake route, backend="cake") | 0.7.3, 0.7.4rc1 | flashinfer.cake_dsa_indexer_det_train (new stable facade, @flashinfer_api; __all__ = cake_dsa_indexer_det_train_topk, cake_dsa_indexer_det_train_topk_workspace_size) | #8936 (merged) |
| flashinfer.dsa_indexer.dsa_indexer_topk(..., backend="cake") | 0.7.3 (entry, default backend='cake'), 0.7.4rc1 (alias of dsa_indexer_det_topk) | flashinfer.cake_dsa_indexer_det_train.cake_dsa_indexer_det_train_topk(...) | #8936 (merged) |
| flashinfer.dsa_indexer.dsa_indexer_topk_workspace_size | 0.7.3 (entry), 0.7.4rc1 (alias) | flashinfer.cake_dsa_indexer_det_train.cake_dsa_indexer_det_train_topk_workspace_size | #8936 (merged) |
| flashinfer.dsa_indexer.dsa_indexer_det_topk(..., backend="cake") | 0.7.4rc1 | flashinfer.cake_dsa_indexer_det_train.cake_dsa_indexer_det_train_topk(...) (same arguments without backend, q_scale, block_table, context_lens, enable_pdl) | #8936 (merged) |
| flashinfer.dsa_indexer.dsa_indexer_det_topk_workspace_size | 0.7.4rc1 | flashinfer.cake_dsa_indexer_det_train.cake_dsa_indexer_det_train_topk_workspace_size | #8936 (merged) |
| flashinfer.experimental.cake_dsa_indexer (backend package) | 0.7.3 (backend), 0.7.4rc1 (previous-path alias package) | flashinfer.experimental.cake.cake_dsa_indexer_det_train | #8936 (merged) |
| flashinfer.experimental.cake_dsa_indexer_det (backend package) | 0.7.4rc1 | flashinfer.experimental.cake.cake_dsa_indexer_det_train | #8936 (merged) |
| cake_backend.dsa_indexer_topk | 0.7.3 (cake_dsa_indexer, real), 0.7.4rc1 (cake_dsa_indexer_det, alias) | cake_backend.cake_dsa_indexer_det_train_topk | #8936 (merged) |
| cake_backend.dsa_indexer_workspace_size | 0.7.3 (real), 0.7.4rc1 (alias) | cake_backend.cake_dsa_indexer_det_train_topk_workspace_size | #8936 (merged) |
| cake_backend.dsa_indexer_det_topk | 0.7.4rc1 | cake_backend.cake_dsa_indexer_det_train_topk | #8936 (merged) |
| cake_backend.dsa_indexer_det_workspace_size | 0.7.4rc1 | cake_backend.cake_dsa_indexer_det_train_topk_workspace_size | #8936 (merged) |
| cake_backend.DSAIndexerRunner | 0.7.3 (real), 0.7.4rc1 (alias) | removed; use cake_backend.DSAIndexerDetRunner (name unchanged since 0.7.4rc1) | #8936 (merged) |
| cake_backend.plan_dsa_indexer_topk | 0.7.3 (real), 0.7.4rc1 (alias) | removed; use cake_backend.plan_dsa_indexer_det_topk | #8936 (merged) |
| cake_backend.prepare_dsa_indexer_topk | 0.7.3 (real), 0.7.4rc1 (alias) | removed; use cake_backend.prepare_dsa_indexer_det_topk | #8936 (merged) |
| cake_backend.validate_dsa_indexer_inputs | 0.7.3 (real), 0.7.4rc1 (alias) | removed; use cake_backend.validate_dsa_indexer_det_inputs | #8936 (merged) |
| tests/experimental/test_cake_dsa_indexer.py, test_cake_dsa_indexer_api.py | 0.7.3, 0.7.4rc1 | tests/experimental/test_cake_dsa_indexer_det_train.py, test_cake_dsa_indexer_det_train_api.py | #8936 (merged) |
| tests/experimental/test_cake_dsa_indexer_fp8.py | 0.7.4rc1 | tests/experimental/test_cake_dsa_indexer_det_train_fp8.py | #8936 (merged) |
| (none) | - | tests/experimental/test_cake_dsa_indexer_det_train_names.py | #8936 (merged) |
| benchmarks/bench_cake_dsa_indexer.py, benchmarks/diag_cake_dsa_indexer_sampling.py | 0.7.3, 0.7.4rc1 | benchmarks/bench_cake_dsa_indexer_det_train.py, benchmarks/diag_cake_dsa_indexer_det_train_sampling.py | #8936 (merged) |
| tests/test_helpers/cake_package.py PACKAGES key cake_dsa_indexer (0.7.3, rc1) / cake_dsa_indexer_det (rc1) | 0.7.3 (cake_dsa_indexer), 0.7.4rc1 (both keys) | cake.cake_dsa_indexer_det_train | #8936 (merged) |
| (none) | - | flashinfer.cake_dsa_indexer_nondet_train (reserved, not added) | planned |

### DSA sparse attention nondet (cake_dsa_sparse_attention_nondet_train)

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| flashinfer.dsa_sparse_attention (module) | 0.7.3, 0.7.4rc1 | flashinfer.cake_dsa_sparse_attention_nondet_train (stable, @flashinfer_api) | #8942 (open) |
| flashinfer.experimental.cake_dsa_train (backend package) | 0.7.3, 0.7.4rc1 | flashinfer.experimental.cake.cake_dsa_sparse_attention_nondet_train (same submodules cake_backend, cake_jit, cake_launch, cake_build) | #8942 (open) |
| dsa_sparse_attention | 0.7.3, 0.7.4rc1 | cake_dsa_sparse_attention_nondet | #8942 (open) |
| dsa_sparse_attention_varlen | 0.7.3, 0.7.4rc1 | cake_dsa_sparse_attention_nondet_varlen | #8942 (open) |
| dsa_sparse_attention_nope | 0.7.3, 0.7.4rc1 | cake_dsa_sparse_attention_nondet_nope | #8942 (open) |
| dsa_sparse_attention_nope_varlen | 0.7.3, 0.7.4rc1 | cake_dsa_sparse_attention_nondet_nope_varlen | #8942 (open) |
| backend DSASparseAttentionFunction | 0.7.3, 0.7.4rc1 | CakeDSASparseAttentionNondetFunction (both rope entries; re-exported by the facade) | #8942 (open) |
| backend DSASparseAttentionNopeFunction | 0.7.3, 0.7.4rc1 | CakeDSASparseAttentionNondetNopeFunction | #8942 (open) |
| backend DSASparseAttentionNopeVarlenFunction | 0.7.3, 0.7.4rc1 | CakeDSASparseAttentionNondetNopeVarlenFunction | #8942 (open) |
| (none) | - | flashinfer.cake_dsa_sparse_attention_det_train (reserved, not added) | planned |

### GQA HD256 nondet (cake_gqa_nondet_train)

| Old name | In | New name | PR |
| --- | --- | --- | --- |
| flashinfer.gqa_train (module) | 0.7.3, 0.7.4rc1 | flashinfer.cake_gqa_nondet_train (stable, @flashinfer_api) | #8939 (merged) |
| flashinfer.experimental.cake_gqa_train (backend package) | 0.7.3, 0.7.4rc1 | flashinfer.experimental.cake.cake_gqa_nondet_train (same submodules cake_backend, cake_jit, cake_build) | #8939 (merged) |
| varlen_gqa_attention (facade + backend) | 0.7.3, 0.7.4rc1 | cake_gqa_nondet | #8939 (merged) |
| varlen_gqa_attention_fwd / backend gqa_attention_fwd | 0.7.3, 0.7.4rc1 | cake_gqa_nondet_train_forward | #8939 (merged) |
| varlen_gqa_attention_bwd / backend gqa_attention_bwd | 0.7.3, 0.7.4rc1 | cake_gqa_nondet_train_backward | #8939 (merged) |
| gqa_train_workspace | 0.7.3, 0.7.4rc1 | cake_gqa_nondet_train_workspace | #8939 (merged) |
| gqa_train_workspace_bytes / backend workspace_bytes | 0.7.3, 0.7.4rc1 | cake_gqa_nondet_train_workspace_bytes | #8939 (merged) |
| backend VarlenGQAAttentionFunction | 0.7.3, 0.7.4rc1 | CakeGQANondetFunction (re-exported by the facade) | #8939 (merged) |
| backend Workspace | 0.7.3, 0.7.4rc1 | CakeGQANondetWorkspace (re-exported by the facade) | #8939 (merged) |
| backend DkvAccSource | 0.7.3, 0.7.4rc1 | CakeGQANondetDkvAccSource (re-exported by the facade) | #8939 (merged) |
| (none) | - | flashinfer.cake_gqa_det_train (cake_gqa_det, CakeGQADetFunction; reserved, not added) | planned |

## Reserved names

- `flashinfer.cake_mhc_head_nondet_train`
- `flashinfer.cake_dsa_indexer_nondet_train`
- `flashinfer.cake_dsa_sparse_attention_det_train`
- `flashinfer.cake_gqa_det_train`
