# CuTe-DSL Kernel Disk Cache

## 1. Motivation

FlashInfer's nvcc-compiled kernels are cached on disk: `JitSpec` compiles a module once into `~/.cache/flashinfer/<version>/<archs>/cached_ops/<name>/<name>.so`, and later processes just `dlopen` it. CuTe-DSL kernels had no disk cache: `cute.compile(..., options="--enable-tvm-ffi")` results were memoized only with `@functools.cache`, so every new process (each benchmark run, test invocation, serving restart) pays the full compilation cost again.

CuTe-DSL's `export_to_c` and `load_module` can serialize and reload a compiled TVM-FFI kernel without recompilation:

```python
compiled = cute.compile(kernel_obj, *fake_args, options="--enable-tvm-ffi")
compiled.export_to_c("kernel.o", function_name="my_kernel")   # persist

module = cute.runtime.load_module("kernel.o", enable_tvm_ffi=True)  # reload
kernel = module.my_kernel   # tvm_ffi.Function, same calling convention
```

`load_module` accepts a raw `.o` directly: it JITLinks the object in memory (no link step, no host toolchain dependency) and returns a `tvm_ffi.Function` with the same calling convention as the freshly compiled kernel.

## 2. Design

The cache deliberately mirrors the existing nvcc kernel cache (`JitSpec`): the same two-level scheme (in-process `functools.cache` + on-disk artifact), the same `cached_ops/` root with one directory per op family, invalidation at the same module granularity as ninja rebuilds, and the same lock conventions. It diverges only where the DSL toolchain forces it — artifacts are single-arch object files rather than multi-arch fatbin `.so`s.

### 2.1 Internal interface for kernel authors

`build_and_load_cute_dsl_kernel` is the CuTe-DSL analogue of `gen_jit_spec()`
on the nvcc side: an internal helper called from FlashInfer's kernel modules,
not part of the public API. Users never interact with the cache directly —
they only observe that CuTe-DSL kernels load in milliseconds after the first
run. A kernel module wires it up as:

```python
from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

@functools.cache                       # level 1: in-process memoization
def _get_compiled_kernel(...):
    kernel_obj = MyKernel(...)         # cheap, no compilation
    return build_and_load_cute_dsl_kernel(
        "nvfp4_quantize",                          # module_name: op family
        f"swizzled_{dtype}_k{K}_sf{layout}_pdl{p}",  # kernel_name: specialization
        lambda: cute.compile(kernel_obj, *fakes, options="--enable-tvm-ffi"),
        extra_key_files=(__file__, helper_module.__file__),
    )
```

On a **hit**, the kernel is JITLinked from the cached `.o` (milliseconds).
On a **miss**, `compile_fn` runs, the result is exported to the module
directory, and the in-process compiled function is returned directly.

### 2.2 Cache on disk layout

CuTe-DSL modules live in `cached_ops/` next to the nvcc modules, one directory per op family:

```
~/.cache/flashinfer/<version>/<archs>/cached_ops/
├── fp4_quantization_100/                 # nvcc module (JitSpec + ninja)
│   ├── fp4_quantization_100.so           #   one multi-function fatbin .so
│   └── build.ninja, *.o
├── nvfp4_quantize_sm100a_cute_dsl/        # CuTe-DSL module
│   ├── meta.json                          #   one per module (invalidation)
│   ├── swizzled_bfloat16_k4096_sf0_pdl0.o #   one .o per specialization
│   └── linear_bfloat16_k4096_sf2_pdl1.o
└── trtllm_gemm/ ...
```

The module directory name is `{module_name}_{arch}_cute_dsl`:

- **`{arch}`** is the DSL's compile target — the `CUTE_DSL_ARCH` environment variable when set, else detected from the current device in the DSL's own format. It must be in the name because a `cute.compile` artifact contains code for exactly one architecture. Note that nvcc modules can omit it when they compile fatbins covering every arch.
- **`_cute_dsl`** marks the toolchain and guarantees no collision with any `JitSpec` URI, following the existing backend-in-suffix convention (`bf16_gemm_cutlass`, `mxfp8_gemm_cutlass`). Nothing in the codebase enumerates `cached_ops/` expecting ninja modules, so co-location is safe, and cache management (`rm -rf ~/.cache/flashinfer/`, `clear_cache_dir()`) covers both kernel types through one root.

The exported TVM-FFI symbol is `{module_name}_{kernel_name}`.

### 2.3 Invalidation

Invalidation is **module-granular**, mirroring ninja's module rebuilds. The per-module `meta.json` records:

| Field | Invalidates when |
|---|---|
| `cute_dsl_version` | nvidia-cutlass-dsl is upgraded |
| `source_sha256` | any file in `extra_key_files` changes (kernel-defining sources) |
| `arch` / `module` | compile target changes (usually also changes the dir name) |

A kernel artifact is valid if and only if its `.o` exists **and** the module `meta.json` matches the expected values. On mismatch the whole module directory is wiped and repopulated lazily. FlashInfer version and nvcc-arch-list changes are already handled by the workspace path.

Hashing whole source files means any edit (even a docstring) invalidates. This is deliberate: it is the same granularity ninja uses, and the alternative (hashing traced MLIR) requires tracing, which defeats the purpose.

### 2.4 Concurrency and crash safety

- A per-module `FileLock` (same `cached_ops/tmp/` convention as `JitSpec`) serializes compilation; the lock is re-checked after acquisition so a process that waited behind a builder loads the fresh artifact instead of recompiling.
- `.o` files are written to a pid-suffixed temp name and committed with atomic `os.replace`, so an `.o` at its final path is always complete.
- `meta.json` is written **after** the first successful export. A crash between the two leaves an `.o` without matching metadata, which reads as a miss and is recompiled/overwritten — never loaded.
- Wiping a stale module while another process has its `.o` mapped is safe on POSIX (the inode survives the unlink).
- Persistence failures degrade gracefully: the freshly compiled in-process kernel is still returned; only the disk write is lost.

`FLASHINFER_CUTE_DSL_DISABLE_CACHE=1` bypasses the cache entirely (always
compile, never touch disk).

## 3. Alternatives considered

### 3.1 Linking one multi-symbol `.so` per module

Full parity with nvcc modules would merge all specializations into a single
`.so`. Rejected: every newly compiled specialization would require re-linking
the `.so`, versioned filenames to dodge `dlopen` staleness, and reload
coordination across processes — high complexity for aesthetics. The
`.o`-per-kernel model is append-only. If AOT distribution is wanted later, an
offline step can link the accumulated `.o`s into one `.so` (the loader
supports both).


## 4. Results from nvfp4_quantize(backend='cute-dsl')

| Metric | Before | After |
|---|---|---|
| Kernel availability in a new process | 0.4–4 s compile per kernel | 3–30 ms JITLink load |
| Artifact size | — | ~28 KB per kernel |
| Steady-state kernel performance | baseline | identical (same binary) |
| Correctness | — | bit-exact vs in-process compile; 5022 tests pass |

The benchmark command
`flashinfer_benchmark.py --routine nvfp4_quantize ... --backends cuda cute-dsl`
compiles on the first run and is compile-free on every subsequent run.

## 5. Limitations and future work

- **Rollout**: only nvfp4_quantize is currently wired up. Other cute-dsl call sites should follow the same pattern: wrap the existing `cute.compile` call in a closure and name the specialization.
- **No `JitSpecRegistry` / AOT integration**: cute-dsl kernels are invisible to `flashinfer aot` and `flashinfer-jit-cache` packaging, and do not honor `FLASHINFER_DISABLE_JIT`. AOT support would prebuild the module directories (or link them into `.so`s, see 3.2).
- **Cross-compile keying**: the arch key mirrors `CUTE_DSL_ARCH`-else-device, but a per-compile `gpu_arch` override passed through `cute.compile` options would not be reflected. No current call site does this.
- **Module-granular wipes**: a single source edit recompiles every cached specialization of that module on next use. Acceptable for development (matches nvcc behavior); irrelevant for deployments with immutable packages.
