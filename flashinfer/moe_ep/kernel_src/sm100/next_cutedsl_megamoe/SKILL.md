# Updating the Blackwell export

Use the reconciled source and exporter revisions recorded in VENDOR.md. Select
both Blackwell inference entry points and retain their runtime dependency
closure. Regenerate the complete `src/sources/` tree, then compare every file
against a second export. Do not combine independently pinned helper snapshots
or modify generated source in FlashInfer. Keep the provider patch set until its
pending changes are available upstream so this revision can be reproduced.

Only `shim/` imports raw sources. Backends consume this package's public API.
Use qualified relative imports so Blackwell, Rubin and the legacy packages
can coexist without replacing each other's top-level modules. Keep shared
staging calls on the existing package API until staging is migrated separately.

Run the Blackwell standard and GenPhase cases in
`tests/moe_ep/test_mega_workspace_knobs.py` and
`tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py`, plus
`tests/moe_ep/test_sm100_standard_features.py` and
`tests/moe_ep/test_sm100_genphase_dtypes.py`. Check weight layouts,
early and late routing weights, masked routes, repeated staging and graph
replay. Preserve GenPhase's default FC2 staging depth and update its tuning
cache revision when kernel behavior or resource use changes.

Check MiniMax and both SiTU forms, live-count changes, deferred outputs,
NVFP4 per-token scales, quantized combine and in-kernel reduction when
changing their workspace bindings. Exercise distributed allocation, peer
communication and graph replay on multiple GPUs after a communication change.

Single-GPU correctness does not establish distributed or performance parity.
Run the existing multirank tests and representative benchmarks before changing
default backend selection. Leave the legacy BF16 paths available throughout.
