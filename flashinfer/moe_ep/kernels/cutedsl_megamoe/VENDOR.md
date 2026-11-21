# CuTe DSL MegaMoE sources

The provider snapshot follows its source layout. Blackwell and Rubin inference
use `sources/`; the retained Blackwell NVFP4 and BF16 implementations and Hopper
use `pre_unification/`. FlashInfer adapters live under
`flashinfer/moe_ep/shims/<sm>/cutedsl_megamoe/`.

```text
cutedsl_megamoe/
├── __init__.py
├── sources/
│   ├── communication/
│   ├── helpers/
│   └── kernel_src/
│       ├── blackwell/inference/{mega,genphase}/
│       ├── rubin/inference/mega/
│       └── schedulers/
└── pre_unification/
    ├── common/
    ├── comm/
    ├── moe_nvfp4_swapab/
    ├── moe_bf16_glu/
    ├── moe_mxfp8_bf16_glu/
    ├── moe_hopper_bf16/
    └── moe_hopper_fp8/
```

## Source and export

The source base is `bdd3f1ee86c8d7d03fe6f0a36533beeea7c4d5fd`. The composed source tree is
`020ada5ef8eed070bfe7121b5e52b117e06fc1d5`; it includes the retained FlashInfer changes described below.
The exporter SHA-256 is `af1e19aaf36495f170dff6a90b2d76b5ebae21adfc390fddf8ad5ddc41e43e13`.
`export_manifest.json` records the entry points, direct adapter requirements,
release-patch checksum and final Python file hashes. Its architecture sections
identify the primary kernel paths; backend capability checks determine which
additional devices an implementation admits.

| Source component | Kernel paths | Adapter |
| --- | --- | --- |
| Blackwell standard and GenPhase | `sources/kernel_src/blackwell/inference/` | `shims/sm100/cutedsl_megamoe/sources/` |
| Retained Blackwell NVFP4 and BF16 | `pre_unification/moe_{nvfp4_swapab,bf16_glu,mxfp8_bf16_glu}/` | `shims/sm100/cutedsl_megamoe/pre_unification/` |
| Rubin standard and GenPhase | `sources/kernel_src/rubin/inference/mega/` | `shims/sm107/cutedsl_megamoe/` |
| Hopper BF16, FP8 and hybrid FP4 | `pre_unification/moe_hopper_{bf16,fp8}/` | `shims/sm90/cutedsl_megamoe/` |

All four components currently use the source base above. Shared runtime files
under `sources/helpers/`, `sources/communication/`, `pre_unification/common/`
and `pre_unification/comm/` have one physical copy and one hash inventory.
The legacy scheduler, epilogue and reducer modules also have consumers outside
their containing kernel directory. A partial refresh must preserve the other
components' pins and check every affected shared-runtime consumer.

From the provider checkout containing the recorded source patches, export the
selection in the manifest. Set `FI_EXPORT_MANIFEST` to this manifest and
`FI_EXPORT_STAGE` to an existing staging directory outside the checkout:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

manifest = json.loads(Path(os.environ["FI_EXPORT_MANIFEST"]).read_text())
destination = Path(os.environ["FI_EXPORT_STAGE"]) / "raw"
subprocess.run([
    sys.executable, "export_src.py", "--kernels", *manifest["kernels"],
    "--modules", *manifest["modules"], "--dst_dir", str(destination),
], check=True)
PY
```

The exporter resolves dependencies, rewrites package imports and generates
lazy entry points. The explicit modules preserve direct adapter requirements,
including fused staging and numerical references. Kernel dependency discovery
cannot infer those callers. The destination must be absent or empty.

Apply the matching release-preparation patch before installation. Its SHA-256
must equal `cleanup_patch_sha256` in the manifest. With `FI_RELEASE_PATCH` set
to that patch's absolute path:

```bash
patch --batch --forward --no-backup-if-mismatch -p1 \
    -d "${FI_EXPORT_STAGE:?}/raw" -i "${FI_RELEASE_PATCH:?}"
```

After release preparation, apply the FlashInfer Blackwell GenPhase patch from
the source export records with the same command, setting `FI_RELEASE_PATCH` to
the GenPhase patch path. In
`sources/kernel_src/blackwell/inference/genphase/token_comm_gen_specialized.py`,
the patch adds `sf_pusher_cta_count` to the positive-value checks in
`GenphaseTokenComm` and removes an obsolete profiling-note reference.

Verify every Python file against the manifest and reject missing or extra
files. The exporter, provider patches and release patch are maintained with
the source export records and are not included in this package. Make kernel
changes there, regenerate the export and run the affected adapter, numerical
and graph tests. Do not edit or format generated kernel files separately.

The two implementation trees retain their own helpers. FlashInfer's Blackwell
and Rubin staging adapters also use `pre_unification/comm/inputs_process.py`.
That is a direct adapter dependency; the provider's `sources/` kernels do not
import the other implementation tree. All imports are package-qualified, so
the exported architectures can coexist without global `common` or `src` aliases.

## Retained integration changes

Blackwell standard and GenPhase preserve FlashInfer's supported quantization
formats, N64 GenPhase geometries, batch-invariant arithmetic and compiled
identities. Routing guards, invalid-route reduction, per-token activation scales,
deferred outputs and persistent finalization remain part of the export.
FC1 writes padded scale columns before FC2 can read them from temporary storage.
The temporary buffer remains caller-owned and is retained across graph replay.
The retained NVFP4 adapter keeps the opt-in NVLS barrier from #9389, with
peer-atomic fallback when multicast mapping is unavailable. Other adapters
retain their existing barrier path.

Rubin retains the integration changes from [FlashInfer #9426](https://github.com/flashinfer-ai/flashinfer/pull/9426):
READY publication after bulk-store completion, scale-padding initialization,
shared-expert fusion, deduplicated combine, GenPhase inbox lifetime and rollover
handling, and batch-invariant arithmetic. Both variants keep their supported
communication settings and separate compiled identities.

Hopper retains FlashInfer's hybrid FP4 path, grouped and deduplicated return,
batched and rotated pulls, segmented readiness and SM90 reduction routines.
Its FP8 FC2 activation scale block is configurable: provider classes default
to 128 channels and the FlashInfer adapter explicitly requests its existing
64-channel contract. Input and weight scale blocks remain 128. The scale choice
also selects the matching reference and heuristic table.

## FlashInfer selectors

SM100's unsuffixed `sm100_nvfp4_nvfp4_bf16_cutedsl` and
`sm100_mxfp8_mxfp8_bf16_cutedsl` selectors use `sources/`. Set
`kernel_variant="genphase"` for GenPhase. The retained NVFP4 implementation
uses `sm100_nvfp4_nvfp4_bf16_cutedsl_legacy`. BF16/BF16 and BF16/MXFP8 remain
under `pre_unification/` with their existing selectors. The older MXFP8 runtime
is replaced by the `sources/` implementation.
The retired MXFP8 snapshot remains available at FlashInfer revision
`9e37595004d839e934a5444b4a71d6026034b136` under
`flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/src/moe_mxfp8_glu/`.
It is not included in this export.

The `_next` selectors and deprecated `nvfp4_cutedsl` / `mxfp8_cutedsl` shorthand
names are removed. Compatible tuning records from the former `_next`
implementation retain their namespace and identity through the rename. Tuning
for the removed legacy MXFP8 implementation does not match the replacement.
Use the replacement's native controls; unsupported legacy scheduler controls
are rejected rather than silently ignored.

SM90 and SM107 selectors keep their existing names. SM120's separately
maintained split and swap-AB drops live under `kernels/flashinfer/` with their
own vendoring records. The backend and adapter checks determine supported SM
capabilities; a source directory name does not admit another device.

## Tuning and qualification

Tuning controls and measurements remain specific to each adapter:
[SM90](../../shims/sm90/cutedsl_megamoe/TUNING.md),
[SM100](../../shims/sm100/cutedsl_megamoe/TUNING.md), and
[SM107](../../shims/sm107/cutedsl_megamoe/TUNING.md).
Source provenance does not imply GPU qualification on every supported device.
Retain each measurement's source revision, workload and timing boundary when
refreshing these records.
