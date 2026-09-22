# SM90 CuTeDSL MegaMoE source provenance

The current implementation combines the FP8 tree merged by #4688
(`9a791d724f382673eec291e018e7eca2f845832e`, final PR head
`c969e2c19f4404a507d9c153cd217d16fe475008`) with the local fused Humming
MXFP4 overlay, on main including the FP8 update from #5338
(`28fae4f2e0ce6b7d9230f828291ca96aa0473ec9`). Both PRs are already in main;
the older pinned base below is historical provenance, not an outstanding
dependency.

## Current local overlay

- Packed Humming E2M1/K32 weights and hybrid FP8 activations share the FP8
  dispatch, scheduling and reduction implementations.
- Guarded paired BF16 FC2 stores, row-address reuse, K256 offset bulk copies
  and zero-count elision preserve the documented completion/reset protocol.
  Optional tail-N8 and segmented-ready strategies have separate narrow domains.
- Main's local 16-byte FC2 store is shared by compatible MXFP4 return modes.
  Its tail-pair scheduler is reused with packed-weight and offset adaptation;
  MXFP4 requires cluster `(1, 2, 1)` and whole-tile readiness. The normal
  MXFP4 tuner includes a bounded set of tail-pair neighbors for
  H7168/I3072/E384/EP4, preserving the prior strategies and default buckets.
  This tuning coverage does not restrict explicit legal tactics on other shapes.
- Tiny positive activation scales retain a finite quantization multiplier
  independently of the rounded dequant scale. Normal/zero arithmetic is kept.
- Shared communication capabilities serve both precisions; FP8's new pair/skip
  defaults stay off. MXFP4's established domain remains, with compact pull and
  training output off. Main's FP8 N8, generate_c and compact-pull behavior stays.
- The current MXFP4 runtime uses fused execution and validates tactics and
  cache identities against the current supported domain. Ordinary library
  CUDA Graph replay is covered separately from direct benchmark launches.
- MXFP4 K128/K256 compute retains its qualified scale-promotion and WGMMA
  wait order; unreachable experimental alternatives and selectors are removed.
- The shared benchmark uses direct CUDA-event launches and the existing
  `--cga M,N` cluster override for both FP8 and MXFP4.

The defaults, weight ABI, numerical contract and current commands are in
`TUNING.md`. Tests cover independent references, full outputs, workspace reuse,
ordinary Graph replay, strategy guards and cache isolation; the PR separately
records which GPU cases and performance comparisons completed. Historical
hashes below identify the pre-overlay snapshot, not the current files. These
local changes do not claim a new verbatim vendor drop or external kernel-team
approval.

## Source chain

The integration was assembled in a dedicated clean kernel staging checkout.
Machine-local paths and non-public remotes are intentionally omitted; the
reproducible source identity is recorded below with base/checkpoint IDs,
package tree IDs, and content hashes.

| Checkpoint | Commit/tree | Purpose |
| --- | --- | --- |
| Original FlashInfer vendor baseline | `fe6ddc6459e6e81cda160b3278786264513692de` | Source of staging commit `9e9b873013756d8c67f79afcae5dd21b8391e149` (tree `0b268ac6638ae227b5ad550b2860e746859736ee`) |
| Pinned FlashInfer PR4688 base | `b1b6b399b7a9885cbe7d543459d0d9a6797b61b4` | Fixed PR snapshot whose FP8 changes are merged into this port |
| Previous MXFP4 fused checkpoint | `1766988168658dbf73f3378467b4224f2fee9875` | Humming fused semantics |
| Previous unified fused/split checkpoint | `d0c99d67efb3a1600a9993377a849ff5f4ed14d8` | Green Context split semantics |
| Historical joint vendor source | package tree IDs and aggregate SHA-256 below | Pinned PR4688 FP8 plus MXFP4 fused/split with fixed production codegen policy |

The joint vendor source was committed atomically at `d80c92d1` in this branch. No
durable staging commit or independently fetchable joint-tree reference is
claimed here; the package tree IDs and content hashes below identify the exact
pre-integration files, and the FlashInfer vendor commit makes those package
trees reachable from the draft PR.

The staging baseline and the two `phase-*` checkpoints above are local staging
objects/refs; this document does not claim that those object IDs are currently
fetchable from the canonical remote. Reviewers can reproduce the target hashes
below from this branch's historical vendor commit, not the edited working
tree. Publishing a retrievable kernel
source ref is a release/provenance follow-up, not something inferred or
fabricated by this integration change.

## Historical atomic package identity (before `fused_local_v1`)

The Git tree IDs identify the four historical package snapshots. Aggregate
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

To reproduce these historical hashes, first extract the vendor commit into a
temporary directory (do not reset the current working tree):

```bash
vendor_audit_dir=$(mktemp -d)
git archive d80c92d1 flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src \
  | tar -x -C "$vendor_audit_dir"
cd "$vendor_audit_dir/flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src"
```

Reproduce the per-package hash from inside an extracted package directory with:

```bash
find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' \
  ! -path './benchmark_data/*' -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum
```

Reproduce the aggregate from the extracted `src/` directory containing the four
packages with:

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

## Updating this tree

For a new vendor drop, form the joint snapshot in a separate clean staging
checkout, replace/compare the four packages together, and record new package
identities. Keep the historical identities above unchanged. Reapply documented
local overlays by their semantics, preserving current main FP8 behavior.

For a local overlay, record its source diff and qualification separately;
do not present the historical hashes as hashes of the edited tree. Recheck
constructor/launch/workspace/reduction interfaces, FP8 and MXFP4 references,
ordinary Graph replay, tuning/cache isolation and direct performance before
publishing updated claims.
