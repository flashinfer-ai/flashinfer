# SM90 CuTeDSL MegaMoE source provenance

Recorded: 2026-09-07 (America/Los_Angeles)

The four directories below form one atomic source snapshot for Hopper FP8 and
Humming MXFP4 x FP8 fused/Green-split MegaMoE. They were assembled and reviewed
in a separate clean kernel staging repository, then copied together into
FlashInfer. They are not a set of independently selected files. The original
snapshot identities below remain historical provenance; the
`fused_local_v1` integration described next is an explicit subsequent change.
Do not replace individual packages with files from unrelated kernel drops.

## Fused MXFP4 integration (2026-09-15 UTC)

`fused_local_v1` integrates the previously isolated paired FC2 peer stores and
address reuse, K256 auxiliary offset bulk copy, guarded zero-count skipping,
optional FC2 N8-tail math, and optional K256 FC1-ready bitmap protocol. It is
not a new verbatim vendor snapshot or an externally approved kernel-team drop.
The eligibility/defaults and 19-field strategy schema are detailed in TUNING.md.

Modified shared paths are compile-time disabled for ordinary FP8 and split.
The host policy in `moe_hopper_fp8/mxfp4_policy.py` is the single implementation
eligibility definition; backend configuration and the tuner use that pure-host
definition without allocating CUDA state. The original package tree IDs and
aggregate hashes below describe the pre-overlay snapshot, not these edited files.

The four-H200 correctness qualification includes ordinary FP8 scale/order
variants, MXFP4 raw-weight cross-rank/sparse/zero-contribution references,
split route/reset stress, graph replay, T2048 full-output comparison, and the
complete 23-candidate offline/shared-online tuning path. Performance acceptance
remains separate; the pull request records completed points and limits. The
policy, candidate expansion and cache checks are included in `tests/moe_ep/`;
GPU coverage includes `test_moe_ep_sm90_pull_mxfp4_mega_multirank.py`. The
integration does not claim a new external kernel-source reference or approval.

## Local follow-up (2026-09-16 UTC)

Two independently qualified changes extend the local overlay; neither replaces
the historical vendor snapshot or claims new kernel-team approval:

- Paired FC2 stores use complete-channel-cluster layout eligibility rather than
  an H7168 equality check. The tail-N8 and segmented-ready protocol domains are
  deliberately unchanged. Four-H200 output and scalar comparisons cover
  H4096/6144/7168/8192. The pull request includes the repeated same-node timing
  results; layout eligibility has pure-host regression coverage.
- Fused Mega MXFP4 activation quantizers retain a finite multiplier for tiny
  positive amax, then independently round the dequant scale. The shared
  epilogue opt-in requires Mega communication, fused execution and MXFP4 mode;
  ordinary FP8, split and non-Mega are excluded. The backend/tuner input path
  also has explicit fused-only opt-in, with a cached post-amax CuTe quantizer.
  Normal output byte equality and tiny-value independent references are covered
  by `test_sm90_mxfp4_safe_quantization.py` and
  `test_moe_ep_sm90_pull_mxfp4_mega_multirank.py`.

Only the qualified implementation was promoted. The formal GPU tests have no
dependency on experimental artifact paths. Consult the pull request for metric
definitions, completed checks and actual clock limitations;
the original package hashes below remain pre-overlay historical identities.

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

## Merge contract

The joint source preserves the pinned PR4688 behavior for ordinary FP8,
including dispatch/combine deduplication, quantized combine, active dispatch
warps, FC1 store offload, early completion publication, producer-warp folding,
and that snapshot's FP8 heuristic table. It adds the prior MXFP4 implementation
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

## Update and acceptance contract

The original drop-refresh contract below remains in force for replacing a
vendor snapshot. The explicitly recorded local overlay above is not such a
replacement; future local changes must preserve its guarded protocol and
refresh its qualification/hash evidence instead of claiming these old hashes.

1. Form any subsequent vendor-drop replacement in a separate clean kernel staging tree.
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
