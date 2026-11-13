# SM107 MegaMoE integration

Keep FlashInfer workspace, staging and tuning changes in `shim/`. Provider
sources are exported under `src/{common,sources}/`, with lazy entry points in
`src/__init__.py`. Use namespaced imports; do not register global `sources` or
`common` modules.

Make kernel changes in the provider before regenerating the export. Preserve
the changes recorded in `VENDOR.md`, apply the recorded release transformations
and verify `export_manifest.json`. Run the affected adapter, numerical and graph
replay tests. Use `TUNING.md` for paired performance comparisons and keep
measurement artifacts outside the source tree.
