# SM90 CuTeDSL MegaMoE source provenance

Recorded: 2026-09-07 (America/Los_Angeles)

The four directories below form one atomic source snapshot for Hopper FP8 and
Humming MXFP4 x FP8 fused/Green-split MegaMoE. They were assembled and reviewed
in a separate clean kernel staging repository, then copied together into
FlashInfer. They are not a set of independently selected files, and they must
not be edited locally for backend, benchmark, or autotune integration.

## Source chain

The integration was assembled in a dedicated clean kernel staging checkout.
Machine-local paths and non-public remotes are intentionally omitted; the
reproducible source identity is recorded below with base/checkpoint IDs,
package tree IDs, and content hashes.

| Checkpoint | Commit/tree | Purpose |
| --- | --- | --- |
| Original FlashInfer vendor baseline | `fe6ddc6459e6e81cda160b3278786264513692de` | Source of staging commit `9e9b873013756d8c67f79afcae5dd21b8391e149` (tree `0b268ac6638ae227b5ad550b2860e746859736ee`) |
| Latest FlashInfer PR4688 base | `b1b6b399b7a9885cbe7d543459d0d9a6797b61b4` | Fixed latest-PR base whose newer FP8 changes are merged into this port |
| Previous MXFP4 fused checkpoint | `1766988168658dbf73f3378467b4224f2fee9875` | Humming fused semantics |
| Previous unified fused/split checkpoint | `d0c99d67efb3a1600a9993377a849ff5f4ed14d8` | Green Context split semantics |
| Current joint source | package tree IDs and aggregate SHA-256 below | Latest PR4688 FP8 plus MXFP4 fused/split with fixed production codegen policy |

The current joint source remains uncommitted until the complete FlashInfer diff
and GPU results are reviewed. No durable staging commit or fetchable joint-tree
reference is claimed here; the package tree IDs and content hashes below identify
the exact files under review. Once accepted, the atomic FlashInfer vendor commit
will make those package trees reachable from the draft PR.

The staging baseline and the two `phase-*` checkpoints above are local staging
objects/refs; this document does not claim that those object IDs are currently
fetchable from the canonical remote. Reviewers can reproduce the target hashes
below directly from this FlashInfer checkout. Publishing a retrievable kernel
source ref is a release/provenance follow-up, not something inferred or
fabricated by this integration change.

## Merge contract

The joint source preserves the latest PR4688 behavior for ordinary FP8,
including dispatch/combine deduplication, quantized combine, active dispatch
warps, FC1 store offload, early completion publication, producer-warp folding,
and the latest FP8 heuristic table. It adds the prior MXFP4 implementation
without replacing those semantics:

- packed Humming E2M1/K32 weights and E8M0 scales;
- hybrid per-routed-row FP8 activation scale handling;
- MXFP4 fused FC1/FC2 and MegaMoE construction;
- Green Context K1/K2 concurrency, K3 join, epoch reset, counter banks, and
  fixed-pointer graph lifecycle;
- invalid `topk_idx == -1` and stale-slot masking in reduction paths.
- eight MXFP4-only experimental codegen selectors fixed to the values used by
  the accepted sweep, so environment state cannot change a public tactic's
  compiled kernel identity.

The highest-risk shared files were merged by role instead of choosing either
side wholesale:

- `src/token_comm.py` retains the latest dedup/group/combine wire protocol and
  the split workspace/communication bodies;
- `moe_nvfp4_swapab/topk_reduce.py` retains the latest `slot_mask` interface
  while accepting the MXFP4 invalid-route mask needed by the fused path;
- FP8/MXFP4 FC12 bases and epilogues retain the latest effective warp-layout
  rules while keeping Humming data and scale paths;
- `src/sym_buffer.py` retains both latest communication regions and split
  lifecycle state.

All FlashInfer-specific validation, candidate policy, persistent caching, and
benchmark plumbing lives outside these packages.

## Atomic package identity

The Git tree IDs identify the four package snapshots under review. Aggregate
SHA-256 is computed from each package with relative filenames included in the
sorted `sha256sum` stream.

| Package | Git tree | Aggregate SHA-256 |
| --- | --- | --- |
| `common` | `0eba1bbc3915b8da74ec453c8783d5c784b4b0d7` | `93ba65862005a7b5a6b057f3aed3a67a96839262063e08fa3912afddce2f6f7b` |
| `src` | `f4b8d7329e08032552c76f821201d11a0e58eebb` | `1bfd2650800fd4e2c6bdc60ba50374b25848be28ebc51675931d606f86a4c954` |
| `moe_nvfp4_swapab` | `727bddc6ee02852628a5fcb1c7c50b1519370ed1` | `19b4d354e70367702d8341c90aaaa6932508220963a87ece4f7457ee7149df24` |
| `moe_hopper_fp8` | `cf4851224a5714984e13d2e9ddb376384b95cd00` | `a1ef52933bd977aa1a8ed0e580ec4f5cbb5755f1b8a42cdf30e7ad5717405662` |

The aggregate across all four named packages is:

```text
ba391b3bf725c3577e692ade79b7e896f3da9969b977c45ed79bd85d51050e00
```

Reproduce the per-package hash from inside a package directory with:

```bash
find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' \
  ! -path './benchmark_data/*' -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum
```

Reproduce the aggregate from the staging repository root with:

```bash
find common src moe_nvfp4_swapab moe_hopper_fp8 -type f \
  ! -name '*.pyc' ! -path '*/__pycache__/*' \
  ! -path 'moe_hopper_fp8/benchmark_data/*' -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum
```

At the 2026-09-07 pre-GPU audit, recursive comparison of each staging package
against its FlashInfer target produced no differences after excluding only
`__pycache__`, `*.pyc`, and `moe_hopper_fp8/benchmark_data`. Those excluded
paths are local generated/raw-data artifacts and are not part of the proposed
source snapshot or its hashes.

## Update and acceptance contract

1. Form any subsequent source change in a separate clean kernel staging tree.
2. Replace and compare all four packages atomically; never patch the copied
   FlashInfer `src/` tree for tuner or backend convenience.
3. Refresh the staged tree, package tree IDs, aggregate hashes, and recursive
   comparison evidence here.
4. Re-audit fused/split constructor, launch, workspace, reduction, and graph
   signatures in the shim.
5. Rerun ordinary FP8 no-regression and MXFP4 standalone, fused/split
   1/2/4-rank, repeated graph, lifecycle, autotune, cache, and locked-H200
   performance gates.

No external kernel-team approval is claimed by this provenance record.
