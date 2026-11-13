# Rubin inference sources

`src/` contains standard and GenPhase inference, with their dependencies in
`src/common/` and `src/sources/`.

```text
next_cutedsl_megamoe/
├── shim/                    # FlashInfer adapters, staging and tuning
├── src/
│   ├── __init__.py           # lazy Rubin entry points
│   ├── common/              # preprocessing, quantization and device helpers
│   └── sources/             # communication, helpers and Rubin kernels
└── export_manifest.json
```

## Source and export

- Source revision: `70a6878de7852ed837d6a4dfc1572728226fbbc4`.
- Shared-runtime revision: `1695115c1fb703f4f1d2ad9076944e4876a3ae01`.
- Source with retained patches: `7a08e1825194461fd17a60adc2fb47ba26af7eb7`,
  recorded as `patch_revision` in `export_manifest.json`.
- Exporter: `export_src.py` at the shared-runtime revision above,
  SHA-256 `3bda4e9bfd5184d9051f3eda17bbbdc0203675e9959459bbc9e2d2629b48d2af`.
- Entry points: `RubinInferenceMegaMoE` and `RubinInferenceGenphaseMegaMoE`.
- Exported Python files: 69, including 11 generated package initializers.

From the provider checkout containing those revisions and patches, export with:

```bash
python export_src.py \
    --kernels RubinInferenceMegaMoE RubinInferenceGenphaseMegaMoE \
    --modules \
        common.cute_ref_ops common.direct_meta common.dsl_helpers \
        common.fc_tile_timeline_host common.host_utils common.inputs_process \
        common.megamoe_constants common.moe_utils common.pipeline_timing \
        common.pipeline_timing_host common.ptx_helpers common.route_reset \
        common.token_segments common.utils \
    --dst-dir "${FI_EXPORT_STAGE:?}/raw"
```

Set `FI_EXPORT_STAGE` to an existing staging directory outside the checkout.
Its `raw/` destination must be absent or empty. The explicit `--modules` list
preserves this drop's shared-runtime selection, including modules used directly
by the FlashInfer shim.

## Export transformations

The exporter resolves dependencies, rewrites imports within the exported package
and generates lazy root entry points. It materializes these source aliases;
paths are relative to the provider checkout and exported `src/`, respectively:

| Provider source | Exported file |
| --- | --- |
| `sources/kernel_src/blackwell/custom_mix_cga_helpers.py` | `sources/kernel_src/rubin/custom_mix_cga_helpers.py` |
| `sources/kernel_src/rubin/inference/local_mega/tma_gather.py` | `sources/kernel_src/rubin/inference/mega/tma_gather.py` |

Apply the release-preparation patch before installing the export. Its SHA-256
must match `cleanup_patch_sha256` in `export_manifest.json`. With
`FI_RELEASE_PATCH` set to that patch's absolute path:

```bash
patch --batch --forward --no-backup-if-mismatch -p1 \
    -d "${FI_EXPORT_STAGE:?}/raw" -i "${FI_RELEASE_PATCH:?}"
```

The manifest hashes describe the files after release preparation. Verify every
file and the complete file list before replacing `src/`. The exporter and
release patch are maintained with the provider export records and are not
included in this package. Keep the export log and file hashes with those records.
Source changes belong in the provider before regeneration; do not edit or format
the installed export. Run the affected adapter, numerical and graph-replay tests.

## Retained changes and integration

The drop retains FlashInfer's route-ID forwarding and invalid-route masking
for separate reduction. GenPhase uses the provider's self-validating combine
and tail protocol, masking invalid scores before reduction. Top-k counts above
the polled reducer's 31-slot limit use the static reducer and wait for peer
stores at the tail barrier.

GenPhase includes native N64 gather support for NVFP4, MXFP8 E4M3 and
MXFP4/MXFP8, retaining the MXFP8 two-CTA cluster option.
Gather padding uses distinct in-bounds source rows. A cluster barrier keeps
peer CTAs alive until shared-memory accesses finish.
The export also retains one-row Gather4 descriptors, counted TMEM
synchronization, invalid-route histogram and scatter guards, and initialization
checks for partial outputs. Standard inference keeps CTA0 reset for small
workspaces on its unsplit, non-deduplicated route. GenPhase waits for the router
before resetting the tail. Native decode scheduling, scale refinement and the
last-arriver/epoch protocol remain in place.

The adapter forwards `use_tanh_sigmoid` and `epilogue_warpgroups`, with defaults
`False` and `1`. Two epilogue warpgroups require standard inference, BF16 bulk
combine and `token_back_mode="epi_warps"`. Both settings stay fixed during
tuning and have separate cache identities.

FlashInfer owns `shim/`, including workspace allocation, weight preparation
and staging extensions. Public selectors and the Blackwell packages retain
their existing locations. See [TUNING.md](TUNING.md) for the measurement protocol.

Standard inference uses a caller-owned temporary buffer, retained by the
adapter across launches and graph replay. GenPhase retains its local-workspace
interface. Shared-expert execution remains disabled by default.
Its provider implementation is included as a dependency of the standard
kernel; FlashInfer does not expose shared-expert parameters in this drop.

Local corrections initialize FC1 scale padding through the last valid writer
and retain 16-byte routed-output alignment. Optional shared-term loads still
require 32 bytes.

`patch_revision` records the source with the review corrections: communication
settings have distinct compiled names, split-CTA FC1 scores follow the selected
routing-weight position, and deduplicated reduction uses 128-bit output stores.
Inference rejects CLC work IDs because its communication and reset protocol
requires a fixed persistent CTA set. The default kernel names are unchanged;
deduplicated reduction has a new output-store cache suffix.
