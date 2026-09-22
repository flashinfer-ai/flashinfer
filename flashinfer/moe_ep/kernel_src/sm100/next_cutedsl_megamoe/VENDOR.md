# Blackwell `next/` MegaMoE source snapshot

## Provenance

- Repository: [bangyus/cutedsl_megamoe](https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe).
- Revision: `e522facf28197c45db443f90693c319c4003abf6`, branch
  `fix/export-tvm-ffi`, based on main
  `e8df888670a44b099e9d7009d5338a4ab45bf848`.
- Exporter: `next/export_src.py`, SHA-256
  `14f23b270f6912e02fab0506bddba45b40d9cecba5936d1d6643908391aa4b90`.
- Selected kernel: `BlackwellInferenceMegaMoE`.
- Payload: 31 implementation modules and nine generated package initializers
  under `src/sources/`. [export_manifest.json](export_manifest.json) records
  the revision, exporter hash, and SHA-256 of every exported file relative to
  `src/`.

The pinned commit adds `tvm_ffi` to the exporter's allowed dependencies and adds
CPU-only regression tests. The preceding main revision rejects the lazy
`tvm_ffi.utils.kwargs_wrapper` import in `helpers/megamoe_aot.py`. No kernel
source changes are part of that fix. This import uses the fix before upstream
review; update the source pin if review changes the exported output.

From an upstream checkout at the pinned revision:

```bash
mkdir -p "${FI_EXPORT_STAGE:?}"
python -I -S next/export_src.py \
    --kernels BlackwellInferenceMegaMoE \
    --dst-dir "$FI_EXPORT_STAGE/sources"
```

The destination must be absent or empty. Copy the generated `sources/` directory
into this package's `src/` without editing it. The exporter resolves the kernel's
dependency closure, rewrites internal imports to relative imports, and generates
package initializers. Compare against an export, not against raw upstream files.

## Scope

This change only ships the new source snapshot. It does not register a backend
or replace `sm100/cutedsl_megamoe`, whose existing runtime behavior is preserved.
There is no runtime API in this package yet.

The export contains upstream Blackwell SiTU activation support and native
MXFP4-weight/MXFP8-activation code. Exposing those through FlashInfer requires a
separate shim and backend change, followed by GPU correctness and performance
validation. Shipping these sources does not establish Kimi K3 support.

The exporter runs with the Python standard library. Executing the exported code
needs compatible CuTe DSL, CUDA bindings and packaging dependencies; its AOT
adapter also uses TVM-FFI. FlashInfer already declares `apache-tvm-ffi>=0.1.11,<0.2`.
This source import does not qualify a runtime toolchain or change dependencies.

## Validation and update policy

Two independent exports at the pinned revision produced identical file lists
and bytes. CPU tests check the source manifest, package import without runtime
dependencies, and wheel contents when `FLASHINFER_TEST_WHEEL` is set. No GPU
execution was performed for this source import.

`src/` is the verbatim exporter output. Do not edit, format, or inject files into
it. Put future FlashInfer adaptations in `shim/` and expose them through this
package's `__init__.py`, following the [shared rules](../../README.md). Backends
must not import `src/` directly. See [SKILL.md](SKILL.md) for the update workflow.

## Pending local diffs versus upstream export

None.
