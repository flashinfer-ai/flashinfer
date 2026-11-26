# Updating the MegaMoE export

Use the source base, composed patches, exporter and release transform recorded
in VENDOR.md. Export the manifest's complete selection, including modules used
directly by FlashInfer adapters. Verify the resulting file list and hashes
against a repeated export before replacing the generated files.

Keep provider changes at their source. Adapters belong under
`flashinfer/moe_ep/shims/<sm>/`; backends own framework dispatch and weights.
Use package-qualified imports and lazy entry points so architectures coexist
without global `common`, `src` or `sources` aliases.

Preserve the retained integration changes in VENDOR.md. Hopper's 128-channel provider default and FlashInfer's explicit
64-channel FC2 activation scales need matching references and heuristic tables. Blackwell's unsuffixed selectors use `sources/`; retained NVFP4 uses
`_legacy`. Preserve compatible tuning identities through path or selector
renames. Revise them when the recorded configuration is no longer compatible.

Run affected numerical and graph-replay tests after a kernel or binding change.
Communication changes need actual multi-GPU execution. Exercise live-count
changes, masked routes, staging tails, deferred and persistent finalization,
in-kernel reduction and supported activation/quantization modes where affected.
Check shared-runtime consumers across architectures, not only the kernel whose
directory changed. Use each SM adapter's TUNING.md for paired benchmarks and
keep raw measurement artifacts outside the source package.
