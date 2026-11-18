# Kimi K3 MegaMoE source export

This tree exports the device closure defined by Kimi commit
`6e8cfe94e9d6c92b184fb46700c72e9f66ccb884` from
[the Kimi repository](https://gitlab-master.nvidia.com/bchislett/kimi-k3-megamoe).
That commit is an overlay. Its required base is TensorRT-LLM
`1.4.0rc3.dev202610300406` from container
`nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc3.dev202610300406`, digest
`sha256:b0c501c715bf11dc098d886194560475936fb2e900994188fc734ba5aceff5b9`.

The source unit is this pinned overlay and its pinned base contract. The exporter
checks the upstream install manifest hash, then its base and overlay hashes before
merging the device files. It does not copy the TensorRT-LLM host integration or installer.
Kimi has no dependency on either current FlashInfer MegaMoE provider.

## Export contract

`scripts/export_kimi_k3_megamoe.py` is the exporter. `export_manifest.json` records
its SHA256, selected input hashes and every generated source hash. `src/` must
match its output byte for byte. Change the exporter, then regenerate; do not edit
`src/` by hand. The maintained API and runtime boundary is `shim/`.

The export starts from `api.py`, `quant_def.py`, the Blackwell MegaMoE kernel and
the two streaming quantizers, then follows relative imports. Parent package
initializers are included; the root initializer exports Blackwell only. The source retains the matching router, scheduler,
workspace layout, FC12 device paths and reducer. The pure tactic constants are
extracted from the pinned overlay's custom-op module without importing that module.

## Pending local diffs vs upstream

All changes below are reproducible exporter transforms:

- Restrict the root initializer to the Blackwell entry point; omit unused Rubin
  kernels from the dependency closure.
- Move `local_expert` initialization before the non-EP16 pull-gather branch guard.
  CuTe DSL 4.8 otherwise joins `None` and `Int32` at the later warp scan.
- Import `OperandMajorMode` from `cutlass.cute.nvgpu.tcgen05`; do not modify global
  CUTLASS modules to supply an alias.
- Replace process-environment reads with a private, context-local option mapping.
  The shim freezes and supplies it during construction and compilation. The adapter
  admits only SM100/SM103, so the production-default architecture predicate is
  constant true and has no cached first-device state.
- Retain quantizer device/JIT code and remove its host cache, scalar launch
  templates, allocating wrappers and global `torch.ops.megamoe` registration.
  The shim owns quantizer compilation and launches on the current stream.
- Fix MXFP8 quantization to the production subnormal mode, `SUBNORMAL_RCP=1`.
- Extract `_TAC_*`, `PROD1_TOKEN_BUCKETS` and `PROD1_DEFAULT_TACTICS` as pure data.

The shim uses its own mask-safe S18 reducer for both combine formats. This differs
from the source's default reducer for small MXFP8 token buckets; generated `src/`
is unchanged.

EP1/EP8 on B300 and EP16 on GB300 passed numerical, eager, graph, masked-route
and empty-rank checks for all four quantization/combine pairs. EP16 used four
nodes with four GPUs each in one NVLink domain. All seven distributed tests
passed on every rank, including coexistence with the standard provider.
That EP16 coexistence run used NVSHMEM 3.7.2 with `NVSHMEM_DISABLE_NVLS=1`
and `NVSHMEM_MAX_TEAMS=512`; Kimi alone needs neither setting.

## Recreate and check

Extract `tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe` and
`tensorrt_llm/_torch/moe/custom_ops/cute_dsl_megamoe_custom_op.py` from the pinned
container into a directory that contains `tensorrt_llm/`. Use an unmodified Kimi
checkout at the pinned commit. From the FlashInfer repository root:

```sh
python scripts/export_kimi_k3_megamoe.py --base /path/to/base --overlay /path/to/kimi
python scripts/export_kimi_k3_megamoe.py --base /path/to/base --overlay /path/to/kimi --check
```

Run the Kimi CPU contract/import tests, build a wheel and test its installed
imports, then run numerical and graph tests on Blackwell. Retain the Apache-2.0
license and notices when updating this drop.
