# Vendored cuDNN Frontend subset: the gated attention block's serving forms
Origin: https://github.com/NVIDIA/cudnn-frontend (`python/cudnn`), commit `fe8b50da0` (cuDNN Frontend 1.32.0).  Import path: `flashinfer.experimental.qwen_3_5_block._cudnn_frost`.
Files: 91 source files (3,896,976 bytes, 75,399 lines) copied from the commit's blobs (`git show fe8b50da0:<path>`; VENDOR_MANIFEST.json holds each file's source sha256 and the vendored file's sha256) and one file from the FE test tree (`sdpa/fwd/_binding_reference.py`, see Native seams), plus the generated `__init__.py`, `_compiled_module.py` (the shim) and these two notes.  Imports rewritten: 297 statements in 45 files (and 26 importlib string references in `engines/manifest.py`).
## License
Every vendored file keeps its original SPDX header verbatim: 55 files `SPDX-License-Identifier: MIT`, 35 files `Apache-2.0`, 1 file `Apache-2.0 AND MIT` (all `Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES`).  FlashInfer's root NOTICE must name this origin and commit; the MIT text ships as `licenses/LICENSE.cudnn-frontend.txt` next to the other third-party texts.  The generated files are FlashInfer-side (Apache-2.0).
## How to re-vendor
```
python vendor.py --fe <cudnn_frontend checkout> --commit <new commit> --closure <closure.json> --target <flashinfer>/flashinfer/experimental/qwen_3_5_block/_cudnn_frost --package flashinfer.experimental.qwen_3_5_block._cudnn_frost --selftest
```
`closure.json` is produced by `tmp/closure/build_closure.py` from the import traces (`trace_closure.py`, run on the host for the declarations and on a cc 10.7 device for compile + execute of each form); re-run both when the FE route changes (a new kernel file, a new lazily imported module).  The self-test must pass before the vendored tree is committed.
## Patches (exact text, applied once each; everything else is byte-identical to the commit)
| id | file | what |
|---|---|---|
| P0 | `gated_attention_block/__init__.py` | the `from .api_bwd import ...` statement and its three `__all__` names are removed (the training half is not vendored) |
| P1 | `engines/manifest.py` | `opt_in_engines_enabled()` returns True: the python engines are offered unconditionally, so `pygraph.validate()` takes the GEMM family's python-native validator and never lowers to the C++ backend graph (the real package's `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` state); the environment variable is neither read nor set |
| P2 | `datatypes.py` | the `importlib.import_module("._compiled_module", package="cudnn")` binding becomes `from . import _compiled_module` (the shim) |
| string sites | `engines/manifest.py` | the `("cudnn._gemm_validate", "validate_graph")`-style module references importlib resolves at runtime are rewritten to the vendored path (the same rule renames that module's logger, `logging.getLogger("cudnn.engines.manifest")`, into the vendored namespace) |
## Excluded (not in the closure; a lazily imported excluded path raises `ModuleNotFoundError` at the call site, never silently)
- `cudnn`: the package root: loads the nanobind extension; the vendored root is written fresh (shim)
- `cudnn._compiled_module`: the nanobind extension (.so): NOT vendored; its Python-visible surface is shimmed
- `cudnn.graph`: the classic graph API facade (graph / jit / graph_cache): the block never uses it
- `cudnn.gated_attention_block.api_bwd`: the training half (backward); dropped from the vendored package __init__ (patch P0)
- the quantized forks (`kernels/quantize*.py`, `proj_gemm_norm_rope_{fp8,mxfp8}.py`, the fp8 / mxfp8 SDPA bodies), the QSA sparse core (`sdpa/fwd/sparse_gqa_sm107.py`, `kernels/qsa_kv_upcast.py`), the unfused norm/RoPE TMA chain (`kernels/qk_norm_rope_tma.py`), the SM80 / SM90 / SM120 prepared hosts and templates, the graph-route planners (`gemm/frost/planning.py`, `sdpa/fwd/placement.py`, `sdpa/fwd/engine.py`, `prepared_staged_*.py`), the e4m3 KV cache path, every backward module.
## Dangling imports (statements rewritten by the prefix rule whose target is outside the closure: excluded paths, reached only by features the serving forms do not declare)
- `engines/manifest.py (string site)`: cudnn._sdpa_validate, cudnn.conv.frost.engine, cudnn.linear_attention, cudnn.linear_attention.graph_analyzer, cudnn.linear_attention.kda_heuristics, cudnn.sdpa.bwd.engine, cudnn.sdpa.bwd.heuristics, cudnn.sdpa.fwd.backend_guard, cudnn.sdpa.fwd.engine
- `gated_attention_block/api.py`: cudnn.sdpa.fwd.sparse_gqa_sm107
- `sdpa/fwd/api_dsl.py`: cudnn._cutlass_compat, cudnn.sdpa.bwd.kernels.thd_helpers, cudnn.sdpa.fwd.kernels.sm80.prepared_host, cudnn.sdpa.fwd.prepared_sm80, cudnn.sdpa.fwd.prepared_sm80_thd, cudnn.sdpa.fwd.prepared_staged_sm80, cudnn.sdpa.packed_copy_sm80
- `sdpa/fwd/engines.py`: cudnn.sdpa.fwd.prepared_sm80
## Native seams (what the serving route reads off the nanobind extension; measured by the import tracer on cc 10.7)
- `_SdpaDenseBinder`: cudnn.sdpa.fwd.api_dsl @ extend.check_support; cudnn.sdpa.fwd.prepared @ decode.compile; cudnn.sdpa.fwd.prepared @ prefill.compile
- `_SdpaThdBinder`: cudnn.sdpa.fwd.api_dsl @ extend.check_support; cudnn.sdpa.fwd.prepared @ extend.compile
- `_read_buffer_sequence`: cudnn.sdpa.fwd.prepared @ decode.execute; cudnn.sdpa.fwd.prepared @ decode.execute2; cudnn.sdpa.fwd.prepared @ extend.execute; cudnn.sdpa.fwd.prepared @ extend.execute2; cudnn.sdpa.fwd.prepared @ prefill.execute; cudnn.sdpa.fwd.prepared @ prefill.execute2
- `backend_graph`: cudnn._pygraph @ decode.compile; cudnn._pygraph @ extend.compile; cudnn._pygraph @ prefill.compile
- `create_handle`: cudnn @ decode.compile; cudnn @ extend.compile; cudnn @ prefill.compile
- `get_stream`: cudnn @ decode.compile; cudnn @ extend.compile; cudnn @ prefill.compile
- `make_operand_buffer`: cudnn.frost.workspace @ decode.execute; cudnn.frost.workspace @ decode.execute2; cudnn.frost.workspace @ extend.execute; cudnn.frost.workspace @ extend.execute2

The shim provides the enums, `cudnnGraphNotSupportedError`, `make_operand_buffer` and `WorkspaceCarve` in Python, and the SDPA binders' `supports_*` capability flags; `backend_graph` / `create_handle` / `get_stream` are unreachable under P1.  The SDPA LAUNCH binders (`_SdpaDenseBinder`, `_SdpaThdBinder`, `_read_buffer_sequence`, `_native_pack_from_facts`) are served IN PYTHON by `flashinfer.experimental.qwen_3_5_block._sdpa_binder` (imported on first use), which builds on `sdpa/fwd/_binding_reference.py` -- cuDNN Frontend's own retired Python frame implementation of the binder, vendored byte-faithfully from the FE test tree (`test/python/sdpa/frost/sdpa_binding_reference.py`, MIT) -- and adds the operand pack and the THD epilogue-gate arm that file predates.  It is the explicit, documented and only binder of this tree (never a silent fallback), for the half (bf16 / f16) plans; a quantized plan is a typed decline.  The extension's sources, for a native binder later: `python/pygraph/{variant_pack.cpp,variant_pack.h,sdpa_dense_binding.cpp,sdpa_thd_binding.cpp,sdpa_fixed_binding.h,sdpa_mxfp8_binding.h}` (nanobind + dlpack only, no cuDNN headers).
## Internal names
Vendored sources are byte-faithful; three files carry hostnames in comments (`proj_gemm_norm_rope.py:56`, `sdpa/fwd/config_sm107.py:178`, `sdpa/fwd/engines.py:1574,2189`) that an FE-side scrub should remove before the FlashInfer PR leaves draft.
