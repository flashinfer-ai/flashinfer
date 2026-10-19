# Vendoring tools for `_cudnn_frost/`

Maintainer tooling, not shipped in wheels (no `__init__.py`): the exporter that produces the sibling `_cudnn_frost/` tree
from a cuDNN Frontend checkout, and the two scripts that compute its closure list.

| file | what |
|---|---|
| `vendor.py` | reads every file of `closure.json` from the pinned commit (`git show <commit>:<path>`), rewrites the `cudnn` import roots to the vendored package, applies the declared patches P0-P6, writes the shim, `VENDOR.md` and `VENDOR_MANIFEST.json`; `--selftest` imports the result with the real `cudnn` package shadowed and declares the three serving forms |
| `closure.json` | the closure list: every vendored file with its module, licence and the sha256 of its blob at the pinned commit, plus the same digest pin for the file vendored from the FE test tree (`extra_files`); the exporter stops on any digest that moved |
| `nanobind_enums.json` | the extension's enum tables and binder capability flags the shim spells; read live from a cuDNN Frontend install when one is importable, else from this copy |
| `build_closure.py` | computes `closure.json` from the import traces plus a static module-level import check (the EXTRA / EXCLUDE policy is at its top) |
| `trace_closure.py` | the import tracer: declares the forms on a host (and compiles + executes them on a cc 10.7 device) and records every module, data file and extension attribute the route touches |

## Re-vendor at a new cuDNN Frontend commit

```
python flashinfer/experimental/qwen_3_5_block/_vendoring/vendor.py \
    --fe <cudnn_frontend checkout> --commit <new commit> \
    --target flashinfer/experimental/qwen_3_5_block/_cudnn_frost \
    --package flashinfer.experimental.qwen_3_5_block._cudnn_frost --selftest
```

The exporter needs `git` and a Python with `torch` and `nvidia-cutlass-dsl` for the self-test (`--python` picks one; no
cuDNN Frontend install is needed).  When a closure file's blob changed, recompute the closure first:

```
python trace_closure.py --form all --mode declare --device cpu  --out traces/host_declare_cpu.json      # any host
python trace_closure.py --form all --mode execute --device cuda --out traces/node_exec_<form>.json      # a cc 10.7 device, per form
python build_closure.py --fe <checkout> --traces traces/ --out closure.json --commit <new commit>
```

Then re-run `vendor.py`, update `README.md` / `NOTICE` where they quote file counts or the pinned commit, and re-run
`tests/experimental/qwen_3_5_block/` on a cc 10.7 device.  Patch P3 (internal names in comments) is dropped once the upstream
files are scrubbed; P4 (the cache-write launcher's one-device check) once the upstream fix is in the pinned commit; P5 / P6 (the
`gemm` package's lazy exports, the sm120 arch family) stay as long as the closure leaves those paths out.
