# Update the Blackwell `next/` MegaMoE export

Use this workflow for this snapshot only. The legacy SM100 snapshot and other
architectures have independent provenance and integrations.

1. Read `VENDOR.md`, `export_manifest.json`, and `../../README.md`. Pin the
   upstream revision and inspect its changes before exporting.
2. Run the upstream CPU regression tests:

   ```bash
   python -S -m unittest tests.test_export_src -v
   ```

3. Export `BlackwellInferenceMegaMoE` to two separate temporary directories
   using the command in `VENDOR.md`. Compare the complete file sets and bytes.
   Resolve exporter or kernel defects upstream, then export again.
4. Replace only this package's `src/sources/` with an unmodified export. Keep
   adaptations outside `src/`. Update the revision, exporter SHA-256, and all
   source hashes in `export_manifest.json`; update `VENDOR.md` with the scope
   and validation evidence for the new revision. File paths in the manifest
   are relative to `src/`.
5. Preserve this tree's exclusions in `.pre-commit-config.yaml` and the ruff
   and mypy configuration. Run the package tests and pre-commit checks from
   the FlashInfer root:

   ```bash
   python tests/moe_ep/test_sm100_next_megamoe_packaging.py -v
   pre-commit run --all-files
   ```

6. Build a wheel using the repository's build workflow. Check its payload:

   ```bash
   FLASHINFER_TEST_WHEEL=/absolute/path/to/flashinfer_python.whl \
       python tests/moe_ep/test_sm100_next_megamoe_packaging.py -v
   ```

The CPU checks establish packaging and source integrity. Once a runtime backend
uses this snapshot, updates also require the affected backend's GPU correctness
and performance checks. Record that evidence separately from export validation.
