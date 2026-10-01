# Updating the Rubin CuTeDSL MegaMoE kernel src

Use this guide when importing a new kernel-team drop or adapting FlashInfer
to changes in that drop. [VENDOR.md](VENDOR.md) records the pinned source and
export exceptions; [TUNING.md](TUNING.md) covers tuning and measurements.

## Layout

```text
kernel_src/sm107/next_cutedsl_megamoe/
├── src/sources/            ← immutable output of upstream next/export_src.py
│   ├── api.py, quant_def.py
│   ├── helpers/, communication/
│   └── kernel_src/         ← Rubin inference/mega plus shared dependencies
├── __init__.py             ← public package API used by FlashInfer backends
├── shim/                   ← FlashInfer-owned adaptation
│   ├── _paths.py           ← exposes src/ as the top-level sources package
│   ├── dependencies.py     ← native SM107 compiler/target checks
│   ├── block_scaled.py     ← config, descriptors, workspace, compile/launch
│   ├── comm.py             ← symmetric allocation, peer mapping, lifetime
│   ├── kernel_helpers.py   ← weight conversion, quantization, Torch reference
│   ├── knob_cache.py       ← heuristic profiles, cache keys and policy
│   ├── autotune.py         ← collective candidate validation and timing
│   └── correctness.py      ← sampled numerical oracle
├── VENDOR.md               ← provenance, supported subset, recorded exceptions
├── SKILL.md                ← this update procedure
└── TUNING.md               ← tuning and benchmark conventions
```

**Keep `src/` byte-for-byte equal to the pinned upstream export.**
Record the exporter revision, selected kernels and generated transformations in
`VENDOR.md`. Fix device/exporter code upstream and re-export; put FlashInfer
adaptation in `shim/` and the backend wrappers.
Preserve the vendored-path exclusions in formatting/lint tools.

## Import layering

Backends → package `__init__.py` → `shim/` → raw `sources` modules.

- Only `shim/` imports raw `sources` modules. Backends and kernel-oracle tests
  use the package API; layer/modes/core use backend APIs.
- Expose helpers needed by backends/tests through the package. Keep heavy
  Torch/CUDA/DSL imports lazy so package discovery does not initialize CUDA.
- `_paths.bootstrap_paths()` points at the sibling `src/`; preserve its guard
  against an unrelated `sources` package already loaded in the process.
  Rubin's `sources` namespace differs from the older SM90/SM100 raw packages.

## When the kernel team drops a new version of src/

### 1. Pin the source and inspect the change

Use a clean upstream checkout at the selected revision. Resolve the complete
SHA and record it before copying; a branch name is not a reproducible pin.
From the upstream repository specified in `VENDOR.md`:

```bash
# Set FI_CUTEDSL_REPO to the upstream checkout and FI_VENDOR_REF to the chosen ref.
git -C "${FI_CUTEDSL_REPO:?}" fetch origin
FI_VENDOR_SHA="$(git -C "$FI_CUTEDSL_REPO" rev-parse --verify "${FI_VENDOR_REF:?}^{commit}")"
git -C "$FI_CUTEDSL_REPO" show --no-patch --format=fuller "$FI_VENDOR_SHA"
```

Compare the old/new inference closure, shared communication/helpers and the
inline sources. If a reference result cites another commit, compare the relevant
file blobs explicitly; equivalence of `rubin/inference/mega/` alone does not
establish equivalence of its dependencies or the full repository.

### 2. Export the selected kernel closure

Use `next/export_src.py` from the pinned commit. It statically follows dependencies,
materializes `COPY_FROM_IMPORT` markers, rewrites imports through package
re-exports and generates a root API for the selected aliases. It does not import
or execute the device kernels and needs only Python's standard library.

From a clean checkout at `FI_VENDOR_SHA`, with an existing scratch staging
parent and an absent/empty output directory:

```bash
python next/export_src.py --list-kernels
python next/export_src.py \
    --kernels RubinInferenceMegaMoE RubinInferenceGenphaseMegaMoE \
    --dst-dir "${FI_EXPORT_STAGE:?}/sources" > "$FI_EXPORT_STAGE/export.log"
sha256sum next/export_src.py
```

The current selection includes the generic and GenPhase inference kernels;
GenPhase is vendored for a separate FlashInfer integration. Select additional
kernels when their FlashInfer integration is included in the update.
The export may include dependencies from other kernel families; record their
source mappings rather than manually removing them.

Review the export before copying it to this drop's `src/sources/`. Replace that
entire directory with the output, including generated initializers, and remove
obsolete modules and `__pycache__` entries. Preserve `shim/` and the package API
outside `src/`. Do not copy `.git`, tester code, build metadata or caches.

Manual closure copying is a fallback when the pinned revision lacks a usable
exporter, or a concrete exporter defect blocks the requested kernel. Record the
reason and exact whole-file mappings in `VENDOR.md`; do not hand-patch exported
source merely to keep the old import paths working. Prefer fixing the exporter
upstream and re-exporting.

### 3. Verify reproducibility and imports

Run the same exporter/selection into a second empty destination and compare file
lists and bytes. Compare the installed `src/sources/` against that output and
preserve SHA-256 hashes with the export log. The current snapshot emits 32 source
modules, nine initializers, three materialized copies and one rewritten import;
these counts describe this pin, not a fixed requirement for later versions.

The exporter checks generated syntax, relative import closure and its external-
module allowlist. Review absolute/dynamic imports and the exporter diff when
updating versions. Confirm raw-kernel imports remain inside `shim/` and that the
FlashInfer package stays lazy. Generated intermediate initializers are empty;
`shim/block_scaled.py` imports the generic kernel through the generated
`sources.RubinInferenceMegaMoE` alias.

### 4. Audit construction, workspace and launch contracts

Symbol names alone are insufficient: requirement dictionaries, argument types,
layouts, ownership and reset semantics can change while names remain stable.

| FlashInfer surface | Upstream contract to compare |
| --- | --- |
| `shim/block_scaled.py`: `_build_kernel()` | `sources/api.py`, `quant_def.py`, and `BlockScaledSwapAbMegaMoeKernel` problem/implementation requirements |
| `Sm107BlockScaledSymmBuffer` allocation and launch resets | Kernel local/shared workspace layouts, token communication, combine buffers/scales and top-k reduction |
| `_runtime_kwargs()`, `cute.compile()` and launch | Kernel call signature, tensor versus pointer arguments, stream, peer mapper and optional normalization |
| `shim/comm.py`, `_peer_mapper()` | `communication/nvlink_domain/symmetric_buffer.py`, rank identity, peer offsets and allocation ownership |
| Config validation and mixed-CGA occupancy | The selected upstream inference solver and host-utils launch recipe, including legal tiles/stages, fallback clusters and reduction constraints |
| `shim/kernel_helpers.py` | Operand layouts, gate/up interleave, quantization/scale encoding and reference numerics |

Preserve the integration's correctness fixes when adapting: physical K-major
weight concatenation, bounded preprocessing temporaries, padded physical router
capacity, per-launch masked-slot clearing/output resets, validated routing and
unit-normalization contract. Recheck warmed graph bindings across different
weights/streams and safe workspace teardown if those contracts change.

### 5. Carry supported changes through the FlashInfer API

New supported knobs may require changes in `Sm107BlockScaledMoeConfig`, both
SM107 backend config dataclasses, workspace construction/pool keys, validation,
and package exports. Inspect both NVFP4 and MXFP8 consumers. An optional upstream
field need not become a public knob unless the integration intends to support it.

Update heuristic profiles, candidate enumeration and cache keys when kernel
semantics or legal configurations change. Bump the backend revision in both
cache lookup and recording when old entries are no longer valid. Require rank
agreement and numerical qualification of tuning winners; do not reuse a cache
across incompatible precision, routing-weight or reduction policies.

`+combine_nvfp4` and `+combine_mxfp8` are future integration work: the current
wrapper selects BF16 combine. Exposing the existing quantized-combine paths
requires configuration, correctly sized payload/scale buffers, reduction and
output handling, cache/pool identities, numerical tests and benchmark variants.
Check those requirements against the upstream kernel interface. Report each
combine format as a separate variant alongside BF16 combine and `+ikr`.

### 6. Validate the changed implementation

For a kernel/shim update, run the affected portable checks and native Rubin
oracle/multirank cases on the claimed formats and EP sizes. The strict runner
rejects skipped/empty suites and records per-rank evidence; select suites for
the changed surface. Example from the FlashInfer repository root, in the
prepared Rubin environment with a fresh persistent results directory:

```bash
export CUTE_DSL_ARCH=sm_107a
python tests/moe_ep/qualify_sm107.py --suite all --world-size 4 \
    --output-dir "${FI_RESULTS:?}/drop-update-ep4"
```

`oracle_sm107` and `mega_sm107` in `tests/moe_ep/run_tests.sh` are alternative
entry points for the same native test files. Choose the suite and EP size for
the changed code; the [validation guide](../../../../../docs/design_docs/moe_ep_sm107_qualification.md)
also has sanitizer and installed-package commands. Isolate negative CUDA tests
and terminate all ranks after a CUDA failure.

Measure affected workloads using `TUNING.md` and the active experiment plan.
Retain absolute latency, raw samples, resolved config, timing/cache scope and
source/environment identity. Existing benchmark numerical and configuration
checks validate measured points. Reuse existing results when the implementation,
environment, and workload are unchanged. Check links and commands for documentation edits.

### 7. Update the record and review the diff

Update `VENDOR.md` with the full upstream SHA, date, closure, any export/local
exceptions and verification evidence. Update `TUNING.md` for changed knobs or
measurement conventions; label inherited performance numbers with their source
revision/environment. Keep test results distinct from planned runs.

Review the mechanical source replacement separately from adapter changes, run
applicable pre-commit checks, and preserve the source-comparison manifest and
validation logs. The package-data rule already includes this directory's
`*.md`; review packaging if the closure introduces new resource types.

## What not to replace with a kernel drop

- `__init__.py` and `shim/` are FlashInfer-owned adapters: adapt them deliberately,
  preserving the public package boundary, rather than overwriting them upstream.
- `backends/mega/kernel/sm107/{nvfp4_nvfp4,mxfp8_mxfp8}_bf16_cutedsl/` contains the
  consumers, not vendored source.
- `core/runtime/bootstrap.py` remains independent of any particular drop;
  preserve process-group/NVSHMEM ownership semantics through the backend API.
- The sibling SM90/SM100/SM120 drops have their own pins and policies; do not
  copy their files into the Rubin closure outside documented upstream exports.
