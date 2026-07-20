# flashinfer.experimental

Fast-moving, arch-scoped kernels, exempt from FlashInfer's stability and
support guarantees. APIs here may change or disappear in any release. The
expected consumers are community containers and local framework forks — not
framework main branches; a kernel needed on a framework release path should
be promoted to core first.

Currently one arch namespace: **`sm12x`** (consumer Blackwell, SM120/SM121),
a curated one-time port of the [b12x](https://github.com/lukealonso/b12x)
CuTe-DSL kernel corpus (`@ 7f7e4f84`). Upstream b12x is now a research
sandbox; this tree is the canonical home.

## Rules (enforced by `tests/experimental/lint/`, blocking in CI)

- **Isolation.** Nothing under `flashinfer/experimental/` imports core
  flashinfer (zero outbound imports — capability gating is vendored in
  `sm12x/_lib/gating.py`), and core never imports experimental. `aot.py`,
  the jit-cache wheel, and the cubin wheel structurally cannot pick up
  experimental code. Tests may import both worlds.
- **Arch restriction.** No SM90/SM100/SM103/SM110 targets anywhere in the
  tree (arch lint).
- **Import topology.** `_lib/` (universal base) ← `<group>/_shared/`
  (intra-group lowering) ← `<group>/<op>/` (one public op). Ops never import
  sibling ops or other groups; anything two groups need moves to `_lib`.
- **Side-effect-free namespace.** `import flashinfer.experimental.sm12x` is
  cheap: no cutlass/triton import, no torch custom-op registration, exactly
  one `FutureWarning` per process (silence with `FLASHINFER_EXP_QUIET=1`).
  Kernels load on first op use.
- **JIT only.** Pure Python + JIT-compiled sources (CuTe DSL, plus raw `.cu`
  via torch cpp_extension inside `comm/pcie/` only). Never AOT.

## One grammar

Every op lives at `sm12x.<group>.<op>` and declares itself via `META`
(an `OpMeta`: entry points, dtypes/recipes, requirements, provenance, test
path). `sm12x.list_ops()` / `sm12x.find_op("moe.fused_moe")` enumerate them;
a registry test keeps `sm12x._OPS` and the op directories in bijection.

Planned (`api_style="planned"`) ops all share the **same shape** — `plan` the
work, size scratch from the plan, `bind` your tensors as views, `run`. Across
three different kernel families, only the module path and the arguments move:

```python
# norm — fused RMSNorm + hyper-connection residual mixing
from flashinfer.experimental.sm12x.norm import mhc

plan    = mhc.plan(mhc.Caps(...))
spec    = plan.scratch_specs()[0]
scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
binding = mhc.bind(plan, scratch=scratch, ...)
residual, post, comb, y = mhc.run_post_pre(..., binding=binding)
```

```python
# moe — fused tensor-parallel routed-expert FFN (weights prepped once per model)
from flashinfer.experimental.sm12x.moe import fused_moe

wplan   = fused_moe.plan_weights(quant_modes="nvfp4",
                                 source_format="modelopt_nvfp4", ...)
experts = fused_moe.prepare_weights(plan=wplan, ...)
plan    = fused_moe.plan(fused_moe.Caps(...))
spec    = plan.scratch_specs()[0]
scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
binding = fused_moe.bind(plan, scratch=scratch, a=x, experts=experts,
                         topk_weights=tw, topk_ids=ti)
out     = fused_moe.run(binding=binding)
```

```python
# attention — MLA decode from compressed KV pages (DeepSeek-V3.2)
from flashinfer.experimental.sm12x.attention import compressed_mla

plan    = compressed_mla.plan(compressed_mla.Caps(...))
spec    = plan.scratch_specs()[0]
scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
binding = compressed_mla.bind(plan, scratch=scratch, q=q,
                              swa_indices=idx, swa_lengths=lens, ...)
out     = compressed_mla.run(swa_k_cache=swa, binding=binding, sm_scale=scale, ...)
```

`plan` is host-side and may allocate; `bind` only narrows/views (never
allocates), which is what makes captured graphs safe; `run*` executes and is
CUDA-graph-capture safe. The family-specific bits vary — a multi-value return
(`mhc`), one-time weight prep (`moe`), cache arguments (`attention`) — while
the plan/scratch/bind/run skeleton does not. One-shot ops
(`gemm.blockscaled.mm`, `quantization.mxfp8.quantize_rows`) are plain
functions; `comm.pcie` collectives are stateful classes (they own IPC
handles). Every op exports `is_supported()`.

Serving controls live at the arch root: warm every kernel shape, then
`sm12x.freeze_kernel_resolution()` so a cache miss raises instead of
compiling inside a live request or graph capture. `sm12x.clear_all_caches()`
clears whatever is already imported.

## Ops

| Op | What it is |
|---|---|
| `attention.paged` | Paged-KV decode/extend (FP8 KV, MSA block-sparse); planner owns tile/split/chunk policy |
| `attention.sparse_mla` | Top-k-selected MLA decode/extend (DeepSeek-V3.2, GLM NSA) |
| `attention.compressed_mla` | MLA decode from compressed KV pages (DSV4), fused merge |
| `attention.nsa_indexer` | NSA index stage: quantize → score → select (+ persistent top-k-2048) |
| `attention.varlen` | Contiguous batched/varlen attention (reduced-assurance tier) |
| `gemm.blockscaled` | One-shot NVFP4/MXFP4/MXFP8 dense block-scaled GEMM |
| `gemm.block_fp8_linear` | DeepSeek-style serialized block-FP8 linear via MXFP8 |
| `gemm.mxfp8_linear` | ModelOpt MXFP8 linear (one-shot) |
| `gemm.wo_projection` | Fused MLA WO-A/WO-B projections (+ inverse-RoPE variant) |
| `moe.fused_moe` | Fused TP routed-expert FFN; recipes nvfp4/mxfp4/w4a8_mx/w4a8_nvfp4/w4a16 |
| `moe.ep_moe` | Expert-parallel W4A16 MoE (cross-rank reduce is the caller's job) |
| `norm.mhc` | Fused RMSNorm + hyper-connection residual mixing + projection |
| `quantization.mxfp8` | BF16/FP16 rows → dense-GEMM MXFP8 layout |
| `comm.pcie` | PCIe collectives: oneshot/DMA all-reduce, FP8 two-shot, DCP A2A (needs nvcc) |

## Environment & caches

All sm12x knobs use the `FLASHINFER_EXP_SM12X_` prefix. Legacy `B12X_*`
variables are read through (copied onto the new names at first use, with a
one-time `DeprecationWarning`). Compile cache:
`FLASHINFER_EXP_SM12X_COMPILE_CACHE_DIR` →
`$FLASHINFER_CACHE_DIR/experimental/sm12x/compile` →
`~/.cache/flashinfer/experimental/sm12x/compile`; disk keys fingerprint the
whole `sm12x/` subtree, so any source edit invalidates exactly.

Known process-global side effect: on first op use, small CUTLASS-DSL runtime
patches apply (compile-only-cache warning suppression, memory-debug no-op,
a `getframeinfo` fast path for 40k-op kernels). Disable with
`FLASHINFER_EXP_SM12X_DISABLE_CUTLASS_RUNTIME_PATCHES=1`; safe alongside a
co-installed b12x. The `torch.ops.flashinfer_sm12x.*` custom-op namespace is
private (torch.compile/graph integration only) — use the Python API.

## Migrating from b12x

Module paths carry the context b12x encoded in names; the verbs are the
closed set `plan / bind / run[_variant] / prepare_* / pack_* / quantize_* /
clear_caches / is_supported` with role classes `Caps / Plan / Binding /
Scratch / Weights`. Representative renames
(`plan.bind(**kw)` ≡ module-level `bind(plan, **kw)`):

| b12x | experimental |
|---|---|
| `b12x_moe_fp4(binding=b)` | `moe.fused_moe.run(binding=b)` |
| `plan_tp_moe_scratch` / `build_tp_moe_fp4_binding` | `moe.fused_moe.plan` / `.bind` |
| `prepare_b12x_fp4_moe_weights` | `moe.fused_moe.prepare_weights` |
| `paged_attention_forward` / `plan_paged_attention_scratch` | `attention.paged.run` / `.plan` |
| `sparse_mla_decode_forward` | `attention.sparse_mla.run_decode` |
| `dense_gemm` | `gemm.blockscaled.mm` |
| `b12x_mhc_post_pre` | `norm.mhc.run_post_pre` |
| `PCIeOneshotAllReduce` | `comm.pcie.OneshotAllReduce` |
| `b12x.freeze_kernel_resolution` | `sm12x.freeze_kernel_resolution` |

Not ported: `attention/mla/legacy/` (retired upstream), the standalone
BF16→NVFP4 TMA quantizer (dead dev harness upstream), b12x's benchmark
harness and exhaustive policy/tuning test sweeps (they remain in b12x).

## Tests

`tests/experimental/` — curated, small-shape, pure-torch-reference tests
(seconds each), per the lightweight-tests rule. The structural lints and
CPU import-hygiene checks run in the `experimental-sm12x` CI workflow; GPU
kernel tests need an SM12x machine:

```bash
pytest tests/experimental/lint -q                  # blocking, stdlib-only
pytest tests/experimental -q                       # full suite (SM120/SM121 GPU)
```
