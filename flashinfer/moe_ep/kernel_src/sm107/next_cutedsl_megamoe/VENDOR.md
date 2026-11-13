# Rubin inference sources

`src/` contains standard and GenPhase inference, with their dependencies in
`src/common/` and `src/sources/`. The source revision is
`70a6878de7852ed837d6a4dfc1572728226fbbc4`, with the shared-runtime changes from
`1695115c1fb703f4f1d2ad9076944e4876a3ae01`.

```text
next_cutedsl_megamoe/
├── shim/                    # FlashInfer adapters, staging and tuning
├── src/
│   ├── __init__.py           # lazy Rubin entry points
│   ├── common/              # preprocessing, quantization and device helpers
│   └── sources/             # communication, helpers and Rubin kernels
└── export_manifest.json
```

The selected entry points are `RubinInferenceMegaMoE` and
`RubinInferenceGenphaseMegaMoE`. `export_manifest.json` records the exported
file hashes and release-transform checksum. Source changes belong in the
provider before regeneration; do not edit the installed export.

The provider's export and release tools are not included in this snapshot.
The cleanup checksum identifies the release transformation. Rebuilding the drop
requires the provider's release process, both entry points above and the shared
modules used directly by the adapter. Verify every exported file against the
manifest before replacing the package, then run the adapter, numerical and
graph-replay tests.

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
