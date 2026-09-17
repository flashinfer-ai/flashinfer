# Blackwell GDN CP-prefill CUDA source

This source export contains CUDA kernels and their TVM FFI host launchers for
context-parallel GDN prefill. `manifest.json` contains only the metadata needed
by the JIT loader: the manifest schema, shared-header paths and hashes, and each
kernel's architecture-specific CUDA source and host binding paths, hashes,
module identifier and entry point.

Shared device utilities live in `cuda/cake_gdn_cp_common.cuh`. The host launchers
encode tensor maps and expose the low-level kernel ABI. The prepared launcher
in `flashinfer.gdn_kernels.blackwell.cake_gdn_cp_backend` handles workspace
allocation, dispatch and graph replay through the public
`flashinfer.gdn_prefill.chunk_gated_delta_rule` API.

The original export snapshot, including historical support contracts, schedule
metadata and the frozen 120-shape performance map, lives in
[`tests/gdn/data/cake_gdn_cp_export_manifest.json`](../../../tests/gdn/data/cake_gdn_cp_export_manifest.json).
It is a historical test fixture, not runtime configuration or an input allowlist.
