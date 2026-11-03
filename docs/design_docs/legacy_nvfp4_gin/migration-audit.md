# Legacy NVFP4 NCCL GIN migration audit

This record is for the engineer implementing the migration.

This targeted import implements the legacy NVFP4 GIN device closure and the
formal FI host path. Numerical acceptance is still in progress; consult
[implementation status](implementation-status.md) for measured outcomes.
The [device hunk ledger](evidence/device-hunk-ledger.json) records imported
files; source comparisons remain reproducible from the fixed objects below.
The external conflict-resolution recipe preserves the initial merge decisions;
later correctness changes, including the generation race fix, supersede its
candidate files and are documented in VENDOR and the investigation.

## Fixed sources

| Role | Commit |
| --- | --- |
| FlashInfer receiver and off control | `e74a1677ab95a1b97aa1c6b6a80b3cf5c0ebac34` |
| Declared vendor snapshot | `fa662f345980f75776ae0c85d2e5c188918ded26` |
| MR !100 base | `41a45039cf81c150f969907dffdd0ba2f10661ca` |
| MR !100 donor | `bc461f87c7b8fdee98fe3655fd2b77fd7819b1e6` |
| MR !100 merge (same donor tree) | `f863d7540614bd546a42867c21adbf51d573061a` |
| Reference checkout HEAD; not the migration source | `dbf44bbd2d1eeccdf30095ef0d9f0a6853b45ea0` |

Both original checkouts were clean. The receiver branch is
`port/legacy-nvfp4-nccl-gin` in `port/flashinfer-nvfp4-gin`; the original
`port/flashinfer` checkout remains the off control. The donor checkout is read-only.

The previously absent vendor object was recovered into the **receiver** object
store from `ssh://git@gitlab-master.nvidia.com:12051/bangyus/cutedsl_megamoe.git`.
Its subject is `Renamed uncombined topk output region to be more descriptive`.
It has the expected legacy `src/`, `common/`, and `moe_nvfp4_swapab/` trees.
The SHA is the one declared by the existing FI vendor record.

The vendor snapshot is **not** an ancestor of MR !100. Their Git merge base is
`6de4469421962013543f63444ad75651edec6ad4`. Do not use either the vendor snapshot
or MR_BASE as an invented common ancestor of the FI files and donor files.

To reproduce the source evidence from this checkout, first restore the objects
if needed, then run the audit:

```bash
git fetch --no-tags --no-write-fetch-head \
  ssh://git@gitlab-master.nvidia.com:12051/bangyus/cutedsl_megamoe.git \
  fa662f345980f75776ae0c85d2e5c188918ded26
git fetch --no-tags --no-write-fetch-head ../cutedsl_megamoe \
  f863d7540614bd546a42867c21adbf51d573061a
python3 scripts/audit_legacy_nvfp4_gin.py
```

The script reads fixed Git objects. It writes
[source-audit.json](evidence/source-audit.json), including the constructor and
launch ABI, hashes and import graph. It also writes reproducible pairwise diffs
under `evidence/comparisons/` (ignored generated files): vendor→FI,
vendor→MR_BASE and MR_BASE→donor. Normalization changes only the two dotted
import prefixes `pre_unification.comm`→`src` and
`pre_unification.moe_nvfp4_swapab`→`moe_nvfp4_swapab`.

## Change ledger

Paths in this table are relative to
`flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/`.

| Receiver change | Provenance and disposition |
| --- | --- |
| `src/src/nccl_gin_layout.py` | Entire new file from MR merge `pre_unification/comm/nccl_gin_layout.py`; byte-identical. No imports need rewriting. |
| `shim/gin.py`: arena partition | Adapted from donor `MegaMoETester.allocate_workspaces()`, preserving transport offset zero and DATA/SF/FP32 weights/shared-workspace order. Only separate BF16 finalization is planned, so no atomic-output arena region. |
| `shim/gin.py`: views | Whole-root, dtype, byte-range and alignment checks; real all-peer GPU reads and lifecycle qualified in 3576298. |
| `shim/gin.py`: LSA deltas | Signed deltas to NCCL window aliases, including self; contiguous same-domain global rank IDs and R<=16. Device mapper imported from fixed donor. |
| `shim/gin.py`: API/topology validation | Actual device API, railed RDMA, N LSA teams and all peers required; no forced-support mode. |
| `shim/gin.py`: control plane | Default-group error/config agreement before each resource and launch boundary; borrowed group never destroyed. UID-derived session identity distinguishes live arenas. |
| `shim/gin.py`: ownership | Drain before window→device comm→comm→root cleanup; stop on failure and retain dependencies. Fatal communication failures require whole-job exit. |
| NVFP4 shim/backend | Lazy compile, one arena, topology/session-aware control validation, on tuning/pooling bypass; off execution and defaults retained. |
| Core layer capability | Default-false collective forward validation; GIN validates every call before rank-local metadata/capacity checks. No raw-source imports. |
| `shim/gin_debug.py` and new tests | Actual fixed-BF16 terms and independent ordered-reduce/Torch oracles; transport/L1 byte snapshots; public relaunch/failure/lifecycle tests. |

## FI patch ledger

These behaviors are retained in the targeted merge. Existing staging and
runner helpers remain byte-identical; modified device files preserve the FI
branches. GPU acceptance is tracked per test and world size in validation.

| FI behavior | Files to resolve during P2/P3 |
| --- | --- |
| Sparse extent, small reset prefix, barrier phase, masked BF16 combine | `src/src/token_comm.py`, NVFP4 `megamoe_kernel.py`, `topk_reduce.py` |
| Custom SwiGLU and unbounded SiTU linear branch | NVFP4 `epilogue_refactor.py`, `kernel_fc12.py`, `megamoe_kernel.py` |
| FC1 per-token scale | Above files plus `custom_ext.py`, communication, shim/backend |
| Singleton expert runtime TMA extent | NVFP4 `kernel_fc12.py`, shim tensor conversion |
| Staging casts, multiplier, DSL compatibility and launch hint | `src/src/inputs_process.py` (not changed by this increment) |
| Triton flat-index guard | NVFP4 `runner_common.py` |
| Custom/persistent finalize, valid-token count, weight placement and caching | Existing shim and backend |

## Dependency and interface audit

The MR scope has 29 changed candidates. The conservative static kernel import
closure has 27 modules, including unchanged dependencies absent from the
candidate list: `sf_swizzle`, `flag_batch`, `sym_buffer`, NVFP4 `contract`,
`dynamic_mainloop`, `moe_persistent_scheduler`, and common constants/math/host
helpers. No `__import__` or `import_module` calls were found in those 27 modules.
This does not prove symbol hooks, external DSL behavior or runtime resources.

`common.direct_meta` and `common.pipeline_timing` are imported by the production
device closure even with diagnostics disabled; they cannot be omitted because
they look like diagnostics. The host timeline helpers and runner are outside
that static kernel closure. Their future inclusion needs an actual consumer.
`relaunch_protocol.py` is outside the static kernel import graph but remains a
protocol/oracle audit input. `common/token_segments.py` and the legacy comm
`token_segments.py` must not be conflated.

The donor adds topology/relay/GIN/config constructor arguments and five GIN
launch operands. FI separately adds custom SwiGLU, per-token scale, persistent
TopkReduce and valid-token launch operands. The exact names are in the JSON ABI
matrix. A successful source merge does not establish layout/launch compatibility.

Both `TokenSrcMetadata` definitions encode one 8-byte word: token in low 32
bits, top-k slot in bits 32–47, global rank in bits 48–63. The donor relay wire
reserves 16 bytes for this record; bytes 8–15 are padding. The N2R2 host test
fixes message offsets and BF16 double-buffer shapes without claiming to test
the device encoder or memory ordering.

## Semantic boundaries and local correction

The constructor/launch ABI unions donor GIN arguments with FI scale, activation
and finalization arguments. Optional GIN handles are erased in off mode. The
registered root starts at the transport segment, then contains activation,
SF, FP32 route weights and shared workspace. Local scratch, weight storage,
indices and separate-reduce output stay outside the registered arena.

The peer mapper is used only for registered same-domain regions. GIN transfers
use offsets relative to that same root, never relative to a subview. Header
route identities keep global source ranks and original top-k slots. Receive
counts, DATA bases and independent SF bases are snapshotted before tail reset;
reference helpers read those snapshots after device synchronization.

The FI small-prefix/sparse optimization is off-only. GIN retains full protocol
participation for partial/empty ranks and distributed tail reset. FC2 BF16 terms
are summed locally in top-k order; a remote node contributes one FP32 sum
rounded to BF16, then origin adds node partials and performs the final cast.
A flat sum is not an equivalent oracle.

The imported token-back deferral had a producer/consumer generation race.
Its fixed target uses the rail generation, stable until tail retirement,
instead of the issue counter that may already have advanced. The issue
publication and tail ordering remain unchanged. This local pending-upstream
diff is recorded in VENDOR and [investigation](investigation.md).

Historical 48/64-experts-per-rank configurations remain unqualified and are
rejected by the fixed 12-per-rank allowlist. This does not equate 48 total
experts in the qualified target to the historical boundary.

## Environment and acceptance

Real donor and receiver correctness runs use qualified GB300 containers on
independent NVLink domains; full versions and job outcomes are in
[validation](validation.md). The login interpreter lacks GPU dependencies.
Host-only checks never substitute for CUDA/NCCL execution.

The user clarified that this feature does not require Rubin/local performance
validation. Those measurements are omitted from this delivery, not reported as
passing. No performance gain is claimed. Final acceptance follows the remaining
correctness and review gates. Keep plan A; no whole-snapshot replacement or new precision path is used.

## User scope clarification

On 2026-11-02 the user stated: “我们的这个feature不涉及rubin环境，可以不用在本地验证”.
Rubin/local performance measurements are therefore not a completion gate for
this delivery. Existing Blackwell correctness checks continue. Performance is
not measured, rather than passed; no hardware restriction is silently bypassed.
