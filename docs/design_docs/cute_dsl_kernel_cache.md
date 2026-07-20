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

The cache shares its lifecycle with the existing nvcc kernel cache through a common abstract base class: `JitSpec` (`flashinfer/jit/core.py`) defines the contract — `try_load()` / `build()` / `load()`, plus the status members `is_compiled` and `get_library_path()` — and implements the shared policy once in a concrete `build_and_load()` template method:

```text
build_and_load()                     JitSpec ABC — shared policy, written once
      │
      ▼
  try_load() ────── hit ────────────►  return cached kernel        (fast path)
      │ miss
      ▼
  FileLock(lock_path)                cross-process lock
      │
      ▼
  try_load() ────── hit ────────────►  return   (another process just built it)
      │ miss
      ▼
  FLASHINFER_DISABLE_JIT set? ─ yes ►  raise MissingJITCacheError
      │ no
      ▼
  build()                            backend-specific compile:
      │                                nvcc: write_ninja + ninja  (host toolchain)
      ▼                                cute-dsl: cute.compile + export_to_c  (GPU)
  load()                             nvcc: dlopen the .so
                                     cute-dsl: in-memory kernel, or JITLink the .o
```

`JitSpecNvcc` (the former `JitSpec` dataclass) implements the abstract methods for nvcc/ninja modules; `JitSpecCuteDsl` implements them for CuTe-DSL kernels; future DSLs (e.g. cutile) follow the same shape.

Callers depend on the interface, not the implementation: `gen_jit_spec()` and the `gen_*_module()` generators keep returning `JitSpec`, and only code that genuinely needs nvcc internals narrows to `JitSpecNvcc` (`build_jit_specs()` rejects non-nvcc specs with a `TypeError`; `JitSpecRegistry` accepts any backend and reports nvcc-only fields with fallbacks, though in practice only nvcc modules register today).

The on-disk conventions also match the nvcc cache: the same two-level scheme (in-process `functools.cache` + on-disk artifact), the same `cached_ops/` root with one directory per op family, invalidation at the same module granularity as ninja rebuilds. It diverges only where the DSL toolchain forces it — artifacts are single-arch object files rather than multi-arch fatbin `.so`s.

One contract nuance: `try_load()` returns the cached artifact only when it is present *and known-valid*, and may conservatively return `None` even when artifacts exist. `JitSpecCuteDsl` decides validity itself (`.o` present + `meta.json` match); `JitSpecNvcc` returns only the AOT artifact, routing the JIT path through `build()` where ninja's dependency scan owns freshness. On a miss, `JitSpecCuteDsl.build()` keeps the freshly compiled kernel in memory and `load()` returns it directly — a build is never followed by a redundant JITLink reload from disk.

### 2.1 Internal interface for kernel authors

`build_and_load_cute_dsl_kernel` is a thin wrapper that constructs a `JitSpecCuteDsl` and runs its `build_and_load()`. It is an internal helper called from FlashInfer's kernel modules, not part of the public API. Users never interact with the cache directly — they only observe that CuTe-DSL kernels load in milliseconds after the first run. A kernel module wires it up as:

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

```text
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

The full precondition set for reusing a cached kernel is spread across the
path, the module metadata, and the artifact filename — every level must
match:

```text
~/.cache/flashinfer/<version>/<nvcc-archs>/cached_ops/   # workspace path:
│                              #   same flashinfer version
│                              #   same nvcc arch list (inherited from the
│                              #   shared workspace; cute-dsl artifacts do not
│                              #   depend on it — changing
│                              #   FLASHINFER_CUDA_ARCH_LIST relocates the
│                              #   cache and conservatively forces recompiles)
└── <module>_<arch>_cute_dsl/  # module dir: same op family +
    │                          #   same DSL compile target (CUTE_DSL_ARCH
    │                          #   else current device)
    ├── meta.json              # module meta: same arch (tripwire), same
    │                          #   nvidia-cutlass-dsl* package stack, same
    │                          #   SHA-256 of the kernel-defining sources
    └── <kernel_name>.o        # filename: identical per-kernel codegen
                               #   parameters (dtype, K, layout, flags, ...)
                               #   — the sole per-kernel key, see Rollout
```

Invalidation is **module-granular**, mirroring ninja's module rebuilds. The per-module `meta.json` records:

| Field | Invalidates when |
|---|---|
| `cute_dsl_version` | any `nvidia-cutlass-dsl*` package changes — the fingerprint covers the whole compiler stack (`nvidia-cutlass-dsl==X;...-libs-base==X;...-libs-cu13==X`), since the codegen backend lives in the libs packages and the CUDA-family variant is encoded in the package *name* |
| `source_sha256` | any file in `extra_key_files` changes (kernel-defining sources) |
| `arch` | never, through normal operation — the compile target is already encoded in the directory name, so a target change routes to a different module dir. Kept as a tripwire against out-of-band directory copies across architectures (a cross-arch `.o` JITLinks fine and would otherwise only fail at kernel launch). |

A kernel artifact is valid if and only if its `.o` exists **and** the module `meta.json` matches the expected values. On mismatch the whole module directory is wiped and repopulated lazily. FlashInfer version and nvcc-arch-list changes are already handled by the workspace path.

The system CUDA toolkit and driver versions deliberately do **not** participate in the key. CUDA-toolkit-tied components absolutely exist in the DSL's IR stack — the NVVM dialects and ptxas itself (statically linked into `_cutlass_ir.so` via the nvPTXCompiler library) come from a specific CUDA release — but they are **vendored inside the `nvidia-cutlass-dsl*` wheels**, so the fingerprint above already tracks the toolkit that actually compiles the kernels. A toolkit upgrade reaches the DSL only as a new package version (or a new `-cuXX` libs package), which changes the fingerprint. `CUDA_HOME`/`CUDA_TOOLKIT_PATH` are referenced by the DSL only in error diagnostics, never in codegen; keying on them would spuriously invalidate on per-shell environment changes while catching no real staleness. (`apache-tvm-ffi` is also excluded: the exported `.o` speaks TVM-FFI's stable C ABI.)

Hashing whole source files means any edit (even a docstring) invalidates. This is deliberate: it is the same granularity ninja uses, and the alternative (hashing traced MLIR) requires tracing, which defeats the purpose.

### 2.4 Concurrency and crash safety

- A per-module `FileLock` (same `cached_ops/tmp/` convention as `JitSpec`) makes each kernel compile **at most once across processes**: the cache is re-checked after acquisition, so a process that waited behind a builder loads the fresh artifact instead of recompiling. The lock is per-module rather than per-kernel — *different* kernels of one op family also serialize their builds — deliberately: `build()` may wipe the whole module directory and rewrite the module-level `meta.json`, which must not interleave with a sibling kernel's export. (Per-kernel compile locks with a module lock only around wipe/meta-commit would allow parallel cold starts; not worth the complexity today.)
- Lock recovery: a crashed holder releases the `flock` automatically (kernel behavior), and a leftover lock *file* is harmless. If a lock is truly wedged (e.g. NFS lock-manager state after a node hang), deleting `cached_ops/tmp/<module>.lock` is safe — artifact commits are atomic and idempotent, so breaking the lock risks at most one duplicated compilation, never a corrupt artifact. `rm -rf ~/.cache/flashinfer/` remains the universal reset.
- `.o` files are written to a pid-suffixed temp name and committed with atomic `os.replace`, so an `.o` at its final path is always complete.
- `meta.json` is written **after** the first successful export. A crash between the two leaves an `.o` without matching metadata, which reads as a miss and is recompiled/overwritten — never loaded.
- Wiping a stale module while another process has its `.o` mapped is safe on POSIX (the inode survives the unlink).
- Persistence failures degrade gracefully: the freshly compiled in-process kernel is still returned; only the disk write is lost.
- Exception contract (uniform across backends, documented on the `JitSpec` ABC): `try_load()` never raises for artifact-level problems — a missing, stale, corrupt, or unloadable artifact logs a warning and reads as a miss, falling through to `build()`; `build()` raises when no usable kernel can result; `load()` raises on failure because it only runs after a successful build.

`FLASHINFER_CUTE_DSL_DISABLE_CACHE=1` disables the **on-disk** layer only:
kernels compile fresh in every new process and nothing is read from or
written to `cached_ops/`. The **in-process** layer (the `@functools.cache`
memoization at call sites, level 1 of the two-level scheme) is unaffected —
within one process each specialization still compiles at most once and is
reused from memory thereafter.

### 2.5 From the former `JitSpec` class to the `JitSpec` ABC

Before this change, `JitSpec` was a single concrete dataclass designed around nvcc compilation: source files + compiler flags in, ninja-built fatbin `.so` out, freshness owned by ninja. None of the existing design is applicable to CuTe-DSL kernel, but the *lifecycle* around it does;  the refactor extracts the lifecycle into an abstract base and leaves each toolchain to its own build model:

```text
JitSpec (ABC)                          # flashinfer/jit/core.py
│    build_and_load()                  # concrete template method (§2 diagram)
│    try_load() / build() / load()     # abstract, per toolchain
│    is_compiled / get_library_path()  # abstract, status reporting
│
├── JitSpecNvcc                        # the former JitSpec dataclass
└── JitSpecCuteDsl                     # flashinfer/jit/cute_dsl_core.py
    (future: JitSpecCutile, ...)
```

The abstract methods, side by side:

| Contract | `JitSpecNvcc` | `JitSpecCuteDsl` |
|---|---|---|
| one instance = | one **module** (many symbols in one `.so`) | one **kernel specialization** (`.o`; op family shares a module dir, `meta.json`, and lock) |
| compilation input | `.cu`/`.cpp` paths + flags (pure data) | live kernel object in a `compile_fn` closure |
| `try_load()` | AOT `.so` only — deliberately ignores the JIT-path `.so`, so `build()` always runs and **ninja judges freshness** | `.o` exists **and** `meta.json` matches → JITLink it; else `None` |
| `build()` | `write_ninja()` + run ninja (no-op if up to date); needs only a host toolchain | wipe module if meta stale, `cute.compile()` (needs a GPU), `export_to_c()`; keeps the compiled kernel in memory |
| `load()` | `dlopen` the `.so` via `tvm_ffi.load_module` | return the in-memory kernel from `build()`, else JITLink the `.o` |
| invalidation authority | ninja dependency scan, at build time | `meta.json` (DSL-stack fingerprint + source SHA-256), at load time |
| `is_compiled` / `get_library_path()` | `.so` exists / `.so` path | `.o` + meta match / `.o` path |

Example — the same lifecycle call, two backends:

```python
gen_jit_spec("fp4_quantization_100", sources, flags).build_and_load()
#   miss → ninja → dlopen fp4_quantization_100.so   (module with many functions)

JitSpecCuteDsl("nvfp4_quantize", "swizzled_bf16_k4096_sf0_pdl0",
               compile_fn, source_sha256).build_and_load()
#   miss → cute.compile → export swizzled_bf16_k4096_sf0_pdl0.o → return in-memory kernel
#   hit  → JITLink the .o (~ms)
```

Remaining notes on the refactor:

- **The ABC keeps the `JitSpec` name**; annotations and `isinstance` checks are undisturbed. Only direct constructions needed the new `JitSpecNvcc` name (one test in-repo).
- **No separate `validate()` method** — folded into `try_load()`, because the backends need opposite control flow (cute-dsl: load-first, validity is a cheap metadata check; nvcc: build-first, only ninja can judge freshness).
- **AOT is outside the ABC contract** for now: `aot_path`/`is_aot` remain nvcc-specific, because a cute-dsl spec holds a compile closure that cannot be enumerated offline (see Limitations).

## 3. Alternatives considered

### 3.1 Linking one multi-symbol `.so` per module

Full parity with nvcc modules would merge all specializations into a single
`.so`. Rejected: every newly compiled specialization would require re-linking
the `.so`, versioned filenames to dodge `dlopen` staleness, and reload
coordination across processes — high complexity for aesthetics. The
`.o`-per-kernel model is append-only. If AOT distribution is wanted later, an
offline step can link the accumulated `.o`s into one `.so` (the loader
supports both).

### 3.2 A standalone cache helper with no shared base class
The initial implementation was a free function that duplicated the lock/double-check/fallback policy alongside `JitSpec` instead of sharing it. It worked, but reviewer discussion converged on the ABC: the policy is written once, `FLASHINFER_DISABLE_JIT` applies uniformly, and future DSL backends (e.g. cutile) get the whole lifecycle by implementing three methods. The standalone helper survives as the thin `build_and_load_cute_dsl_kernel` wrapper over `JitSpecCuteDsl`.

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

- **Rollout**: only nvfp4_quantize is currently wired up. Other cute-dsl call sites should follow the same pattern: wrap the existing `cute.compile` call in a closure and name the specialization. The kernel name is the **sole per-kernel cache key** (`meta.json` guards only module-wide facts), so a name function must encode every codegen parameter — new adopters should replicate the naming tests in `tests/jit/test_cute_dsl_cache.py` (signature coverage of the kernel getters + per-argument perturbation) for their own name functions.
- **No `JitSpecRegistry` / AOT integration**: the registry now *accepts* any `JitSpec` backend, but cute-dsl kernels do not register themselves yet (only `gen_jit_spec()` registers), so they are invisible to the `flashinfer` CLI's module listing and to `flashinfer aot` / `flashinfer-jit-cache` packaging. (`FLASHINFER_DISABLE_JIT` *is* honored — it comes free from the shared `build_and_load()` template method.) AOT support would additionally require the spec to become declarative data (a factory reference + parameters) instead of a compile closure, so specs can be enumerated and prebuilt offline; then prebuild the module directories or link them into `.so`s (see 3.1).
- **Cross-compile keying**: the arch key mirrors `CUTE_DSL_ARCH`-else-device, but a per-compile `gpu_arch` override passed through `cute.compile` options would not be reflected. No current call site does this.
- **Single-target-arch assumption (heterogeneous multi-GPU)**: the cache key records what `cute.compile` actually compiles for — the DSL's resolved target (`CUTE_DSL_ARCH`, else the *current* device). Compiling for a non-current device in a mixed-arch process is not supported: it would require steering the compilation itself (per-compile `gpu_arch`) together with the key, not just parameterizing the key. The in-process `@functools.cache` level has the same current-device assumption today, so the disk cache does not regress multi-GPU behavior.
- **Module-granular wipes**: a single source edit recompiles every cached specialization of that module on next use. Acceptable for development (matches nvcc behavior); irrelevant for deployments with immutable packages.
