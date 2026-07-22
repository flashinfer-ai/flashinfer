# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashInfer is a GPU kernel library for LLM serving that uses **JIT (Just-In-Time) compilation by default**. This means kernel code changes are automatically picked up without reinstalling the package - extremely convenient for development.

## Quick Reference

| Task | Command |
|------|---------|
| Install for development | `pip install --no-build-isolation -e . -v` |
| Initialize submodules | `git submodule update --init --recursive` |
| Install CUPTI for benchmarking | `pip install -U cupti-python` |
| Run all tests | `pytest tests/` |
| Run specific test | `pytest tests/path/test_file.py::test_function` |
| Run multi-GPU test | `mpirun -np 4 pytest tests/comm/test_allreduce_unified_api.py` |
| Run benchmark | `python benchmarks/flashinfer_benchmark.py --routine <name> <flags>` |
| Run linting | `pre-commit run -a` |
| Install pre-commit hooks | `pre-commit install` |
| Clear JIT cache | `rm -rf ~/.cache/flashinfer/` |
| Enable API logging (basic) | `export FLASHINFER_LOGLEVEL=1` |
| Enable API logging (detailed) | `export FLASHINFER_LOGLEVEL=3` |
| Enable API logging (with stats) | `export FLASHINFER_LOGLEVEL=5` |
| Set API log destination | `export FLASHINFER_LOGDEST=mylog.txt` |
| Enable verbose JIT logging | `export FLASHINFER_JIT_VERBOSE=1` |
| Enable debug build | `export FLASHINFER_JIT_DEBUG=1` |
| Set target architectures | `export FLASHINFER_CUDA_ARCH_LIST="8.0 9.0a"` |
| Set parallel compilation | `export FLASHINFER_NVCC_THREADS=4` |
| Limit parallel ninja jobs | `export MAX_JOBS=4` |
| Enable GDN native short-T path | `export FLASHINFER_GDN_WY_NATIVE_T=1` |
| Enable GDN strided QKV path | `export FLASHINFER_GDN_WY_STRIDED_QKV=1` |
| Enable GDN native A/B tensors | `export FLASHINFER_GDN_WY_NATIVE_AB=1` |

## Quick Start for Development

### Installation

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
pip install --no-build-isolation -e . -v
```

**Important**: The `--recursive` flag is required to initialize submodules in `3rdparty/` (cutlass, spdlog).

If you forgot `--recursive` when cloning:
```bash
git submodule update --init --recursive
```

That's it! You can now:

- Run all benchmarks and unit tests
- Modify kernel source code in `include/` without reinstalling
- Changes are JIT-compiled on next use

The `--no-build-isolation` flag prevents pip from pulling incompatible PyTorch/CUDA versions from PyPI.

### How JIT Compilation Works

When you call a FlashInfer API:

1. **First call**: Generates specialized CUDA code based on parameters (dtype, head_dim, etc.), compiles it with ninja, caches the .so file
2. **Subsequent calls**: Uses cached compiled module
3. **After kernel changes**: Automatically detects changes and recompiles

**No manual rebuild step needed** - just edit `.cuh` files and run your code again.

### Pre-compiled Packages (Optional)

FlashInfer provides optional pre-compiled packages for users who want faster initialization:

- `flashinfer-jit-cache`: Pre-built kernel cache
- `flashinfer-cubin`: Pre-compiled kernel binaries

**For development, you typically DON'T need these.** JIT compilation is fast enough and gives you live code reload.

## Testing

Run all tests:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/attention/test_hopper.py
```

Run specific test function:

```bash
pytest tests/attention/test_hopper.py::test_single_prefill
```

### Skipping Tests Based on CUDA Architecture

Use `flashinfer.utils` functions to skip tests on unsupported GPU architectures:

**Available check functions:**
- `get_compute_capability(device)` - Returns `(major, minor)` tuple
- `is_sm90a_supported()` - Hopper (requires CUDA 12.3+)
- `is_sm100a_supported()` - Blackwell (requires CUDA 12.8+)
- `is_sm100f_supported()` - Blackwell feature-set (requires CUDA 12.9+)
- `is_sm110a_supported()`, `is_sm120a_supported()`, `is_sm120f_supported()`, `is_sm121a_supported()`

**APIs decorated with `@backend_requirement`** also provide:
- `api_name.is_compute_capability_supported(cc)` - e.g., `mm_fp4.is_compute_capability_supported(100)`
- `api_name.is_backend_supported("backend")` - e.g., `mm_fp4.is_backend_supported("cudnn")`
- `api_name.is_backend_supported("backend", cc)` - e.g., `mm_fp4.is_backend_supported("cudnn", 80)`

**Example:**
```python
from flashinfer.utils import is_sm90a_supported

def test_hopper_attention():
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("Requires SM90a")
    # Test code...
```

**Common requirements:**

| Feature | Min SM | Check Function |
|---------|--------|----------------|
| FlashAttention-3 | SM90a | `is_sm90a_supported()` |
| MLA Attention | SM100a | `is_sm100a_supported()` |
| FP8 GEMM | SM89+ | `get_compute_capability()[0] >= 9` |

**Note:** `tests/conftest.py` auto-skips tests that trigger OOM, but tests should be written to avoid OOM by using appropriate problem sizes.

## Benchmarking

FlashInfer provides a unified benchmarking framework in `benchmarks/flashinfer_benchmark.py`.

**Key features:**
- Supports attention, GEMM, and MOE kernels
- Multiple backends: FlashAttention2/3, cuDNN, CUTLASS, TensorRT-LLM, cuBLAS
- **CUPTI timing (recommended)**: Hardware-level profiling for accurate GPU kernel time
  - Automatically falls back to CUDA events if CUPTI unavailable
  - Install: `pip install -U cupti-python` (requires CUDA 13+)
- Batch testing, reference checking, CSV output

**Quick example:**
```bash
python benchmarks/flashinfer_benchmark.py \
    --routine BatchDecodeWithPagedKVCacheWrapper \
    --backends fa2 cudnn \
    --batch_size 32 --s_kv 2048 \
    --num_qo_heads 32 --num_kv_heads 8 \
    --head_dim_qk 128 --head_dim_vo 128 \
    --page_size 16 --refcheck -vv
```

**Python API:**
```python
from flashinfer.testing import bench_gpu_time

# CUPTI preferred, auto-fallback to CUDA events
median_time, std_time = bench_gpu_time(
    my_kernel, args=(x, y), enable_cupti=True, num_iters=30
)
```

→ **For complete benchmarking guide, see [`.claude/skills/benchmark-kernel/skill.md`](.claude/skills/benchmark-kernel/skill.md)**

## Code Linting

Run all pre-commit hooks:

```bash
pre-commit run -a
```

Install hooks to run on every commit:

```bash
pre-commit install
```

## Code Review

When reviewing a diff (as an agent or a human), follow the shared focus areas, kernel-review
policy, and effort calibration in [`docs/code_review_guidance.md`](docs/code_review_guidance.md).
Note: unlike human review, agents keep **kernel implementation details in scope** — read the
kernel logic and report bugs, labeling findings by confidence.

→ **For the complete review rule set, see [`docs/code_review_guidance.md`](docs/code_review_guidance.md)**

## Architecture: JIT Compilation System

FlashInfer's JIT system has three layers:

### Layer 1: JitSpec (flashinfer/jit/core.py)

`JitSpec` is an abstract base class defining the kernel-module lifecycle
(`try_load()` / `build()` / `load()`, with the shared `build_and_load()`
template method handling caching, locking, and `FLASHINFER_DISABLE_JIT`).
One subclass per compilation toolchain:

- `JitSpecNvcc` (nvcc/ninja modules, returned by `gen_jit_spec()`) defines:
  - `name`: Unique identifier (URI hash from parameters)
  - `sources`: List of .cu/.cpp files to compile
  - `extra_cuda_cflags`, `extra_cflags`, `extra_ldflags`: Compiler flags
- `JitSpecCuteDsl` (flashinfer/jit/cute_dsl_core.py) caches CuTe-DSL kernels
  (see "CuTe-DSL kernels" under Module Caching below)

### JIT Directory Rules

**NEVER write to package directories** - they may be read-only after installation.

| Directory | Writable | Use for |
|-----------|----------|---------|
| `FLASHINFER_GEN_SRC_DIR` | ✓ Yes | Generated source files (Jinja output, copied .cu files) |
| `FLASHINFER_JIT_DIR` | ✓ Yes | Compiled `.so` outputs |
| `FLASHINFER_CSRC_DIR` | ✗ No | Read-only source templates |
| `FLASHINFER_AOT_DIR` | ✗ No | Read-only pre-compiled binaries |

### Compilation Context: Architecture-Specific Compilation

FlashInfer uses `CompilationContext` to manage CUDA architecture targets. Some kernels only work on specific GPU architectures (e.g., Hopper SM90, Blackwell SM100/SM12x).

**How it works:**
- Auto-detects GPUs in system or reads `FLASHINFER_CUDA_ARCH_LIST` environment variable
- JIT modules specify `supported_major_versions=[9, 10, 11, 12]` to limit compilation to specific SM versions
- If GPU not supported → `RuntimeError: No supported CUDA architectures found`

→ **See [`.claude/skills/add-cuda-kernel/skill.md`](.claude/skills/add-cuda-kernel/skill.md) for usage examples**

### Layer 2: Code Generation

Every `gen_*_module()` function in `flashinfer/jit/` follows this pattern:

```python
def gen_some_module(dtype_in, dtype_out, ...):
    # 1. Compute unique identifier from parameters
    uri = get_some_uri(dtype_in, dtype_out, ...)

    # 2. Create generation directory
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri

    # 3. (Optional) Render Jinja template to generate type-specialized config
    # Skip this step if you don't need type specialization
    with open(jit_env.FLASHINFER_CSRC_DIR / "some_customize_config.jinja") as f:
        template = jinja2.Template(f.read())
    config_content = template.render(
        dtype_in=dtype_map[dtype_in],
        dtype_out=dtype_map[dtype_out],
        # ... more parameters
    )
    write_if_different(gen_directory / "some_config.inc", config_content)

    # 4. Copy source files to gen directory
    sources = []
    for fname in ["some_kernel.cu", "some_jit_binding.cu"]:
        shutil.copy(jit_env.FLASHINFER_CSRC_DIR / fname, gen_directory / fname)
        sources.append(gen_directory / fname)

    # 5. Return JitSpec
    return gen_jit_spec(uri, sources, extra_cuda_cflags=[...])
```

**Note**: If your operation doesn't need type specialization, you can skip step 3 entirely and just copy the source files directly.

### Layer 3: Compilation and Loading

`JitSpecNvcc` methods:

- `write_ninja()` - Generates `build.ninja` file
- `build()` - Executes `ninja` to compile sources
- `build_and_load()` - Compiles and loads via TVM-FFI

The generated `build.ninja` file uses nvcc to compile .cu → .cuda.o → .so, then loads via TVM-FFI.

### Jinja Templates (Optional)

**Note: Jinja templates are NOT required.** You can write C++ code directly without templating.

For operations that need type specialization, templates in `csrc/*.jinja` can generate C++ code:

```jinja
// Input template
using DTypeIn = {{ dtype_in }};
using DTypeOut = {{ dtype_out }};
constexpr int PARAM = {{ param_value }};

// After render
using DTypeIn = float16;
using DTypeOut = float16;
constexpr int PARAM = 128;
```

This allows the same CUDA template code to be compiled with different concrete types. However, if your operation doesn't need this, you can skip Jinja and write the `.cu` files directly.

## Directory Structure

```
flashinfer/
├── include/flashinfer/           # Header-only CUDA kernel templates
│   ├── attention/                # Attention kernels
│   ├── gemm/                     # GEMM kernels
│   ├── comm/                     # Communication kernels
│   ├── mma.cuh                   # Matrix multiply utilities
│   ├── utils.cuh                 # Common utilities
│   └── [...]
│
├── csrc/                          # Framework bindings (via TVM-FFI)
│   ├── *.cu                       # Kernel launcher implementations
│   ├── *_jit_binding.cu           # TVM-FFI exports
│   ├── *_customize_config.jinja   # Type config templates (optional)
│   └── [...]
│
├── flashinfer/                    # Python package
│   ├── jit/
│   │   ├── core.py                # JitSpec, compilation infrastructure
│   │   ├── cpp_ext.py             # Ninja build generation
│   │   ├── env.py                 # Workspace paths
│   │   ├── attention/             # Attention module generators
│   │   ├── gemm/                  # GEMM module generators
│   │   ├── fused_moe/             # MOE module generators
│   │   └── [...]
│   ├── gemm/                      # GEMM Python APIs
│   ├── fused_moe/                 # MOE Python APIs
│   ├── comm/                      # Communication Python APIs
│   ├── *.py                       # Other high-level Python APIs
│   ├── aot.py                     # AOT compilation for pre-built packages
│   └── [...]
│
├── tests/                         # Test suite
│   ├── attention/                 # Attention kernel tests
│   ├── gemm/                      # GEMM kernel tests
│   ├── moe/                       # MOE kernel tests
│   ├── comm/                      # Communication tests
│   ├── utils/                     # Utility tests
│   └── conftest.py                # Pytest configuration
│
└── build_backend.py               # PEP 517 build backend
```

### Critical Rule: Framework Separation

**Torch headers MUST NOT be included in `include/` directory files.**

- `include/`: Framework-agnostic CUDA kernels (accept raw pointers)
- `csrc/`: Framework bindings via TVM-FFI (currently PyTorch, but can support other frameworks)

## Adding a New Operation

→ **For complete step-by-step tutorial, see [`.claude/skills/add-cuda-kernel/skill.md`](.claude/skills/add-cuda-kernel/skill.md)**

**Quick overview of the process:**
1. Write kernel in `include/flashinfer/new_op.cuh` (framework-agnostic, raw pointers)
2. Write launcher in `csrc/new_op.cu` (PyTorch tensor handling)
3. Create TVM-FFI bindings in `csrc/new_op_jit_binding.cu`
4. (Optional) Create Jinja template for type specialization
5. Write JIT module generator in `flashinfer/jit/new_op.py`
6. Write Python API in `flashinfer/new_op.py` with `@functools.cache`
7. Write tests in `tests/`
8. Register in `flashinfer/aot.py` for AOT compilation
9. Export in `flashinfer/__init__.py`
10. Add a `TraceTemplate` in `flashinfer/trace/templates/` and wire it via `@flashinfer_api(trace=...)` (see below)
11. Add an example call in `tests/trace/example.py`, re-run to regenerate `fi_trace_out/`, and commit the new JSON files

### Trace Template Checklist (for new or updated APIs)

Every public API decorated with `@flashinfer_api` should also carry a `trace=` argument so that `fi_trace()` works and auto-dump produces a benchmark definition JSON.

1. **Create or update a `TraceTemplate`** in `flashinfer/trace/templates/<category>.py` (e.g., `norm.py`, `activation.py`, `cascade.py`, `gdn.py`). Define `axes`, `inputs`, `outputs`, and optionally a `reference` function.
2. **Wire the template** to the API: `@flashinfer_api(trace=my_trace)` on the Python function (or class method's `run()`).
3. **Add an example call** in `tests/trace/example.py` that exercises the new trace with realistic shapes.
4. **Regenerate examples**: `rm -rf tests/trace/fi_trace_out && python tests/trace/example.py` — verify the expected JSON appears.
5. **Update the docstring** in `tests/trace/example.py` to list the new file(s).
6. **Run tests**: `pytest tests/trace/ -v` — all template-consistency and end-to-end tests must pass.
7. **Commit the new JSON files** under `tests/trace/fi_trace_out/` alongside the code changes.

**Example implementations:**
- **Simple**: `flashinfer/norm/__init__.py` (RMSNorm) - no Jinja, good starting point
- **Moderate**: `flashinfer/sampling.py` - with Jinja templating
- **Complex**: `flashinfer/decode.py` - plan-run pattern, advanced workspace

## Key Architectural Patterns

### Module Caching

FlashInfer uses two-level caching to avoid recompilation:

1. **Python-level** (`@functools.cache`): In-memory cache of loaded modules
2. **File-level** (`~/.cache/flashinfer/`): Compiled `.so` files on disk

**Cache invalidation** (automatic):
- Source file changes (SHA256 hash)
- Compilation flags change
- CUDA architecture change
- FlashInfer version change

URI computed as: `hash(operation_type + parameters + source_hashes + flags + cuda_arch)`

**Cache management:**
- Clear cache: `rm -rf ~/.cache/flashinfer/`
- Override location: `export FLASHINFER_WORKSPACE_BASE="/scratch"`

**CuTe-DSL kernels** (`flashinfer/jit/cute_dsl_core.py`): `cute.compile()` has no
persistent cache, so `JitSpecCuteDsl` (a `JitSpec` subclass, wrapped by the
`build_and_load_cute_dsl_kernel()` helper) exports compiled kernels as
object files (`export_to_c()`) and reloads them with `cute.runtime.load_module(...,
enable_tvm_ffi=True)`. Artifacts live in `cached_ops/` next to the nvcc modules, one
directory per op family — `cached_ops/<module>_<arch>_cute_dsl/` (e.g.
`nvfp4_quantize_sm100a_cute_dsl/`) holding one `meta.json` plus one `.o` per
specialization. The arch comes from the DSL's compile target (`CUTE_DSL_ARCH` or the
current device) since the artifacts are single-arch, unlike nvcc fatbins.
Invalidation is module-granular: a changed nvidia-cutlass-dsl version or
kernel-source SHA256 wipes and lazily rebuilds the module. Reference usage:
`flashinfer/quantization/kernels/nvfp4_quantize.py`. Disable with
`FLASHINFER_CUTE_DSL_DISABLE_CACHE=1`.

### Dispatch Macros

Handle combinatorial parameter spaces:

```cpp
DISPATCH_DTYPE(input_dtype, DTypeIn, {
  DISPATCH_DTYPE(output_dtype, DTypeOut, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      LaunchKernel<DTypeIn, DTypeOut, BLOCK_SIZE>(...);
    });
  });
});
```

Defined in `.jinja` files and expanded after rendering.

## API Logging with @flashinfer_api

FlashInfer provides the `@flashinfer_api` decorator for debugging API calls.

**Key features:**
- **Crash-safe**: Logs inputs BEFORE execution (preserves info even if kernel crashes)
- **Zero overhead when disabled**: `FLASHINFER_LOGLEVEL=0` (default)
- **Multiple verbosity levels**: 0 (off), 1 (names), 3 (inputs/outputs), 5 (+ statistics)
- **CUDA graph compatible**: Auto-skips stats during graph capture

**Quick usage:**
```bash
# Enable detailed logging
export FLASHINFER_LOGLEVEL=3              # 0, 1, 3, or 5
export FLASHINFER_LOGDEST=debug.log       # stdout, stderr, or file path

python my_script.py
```

**Why use this?**
- Debug CUDA crashes (see inputs that caused crash)
- Track tensor shapes/dtypes through pipeline
- Detect NaN/Inf issues (level 5)

→ **For complete debugging guide, see [`.claude/skills/debug-cuda-crash/skill.md`](.claude/skills/debug-cuda-crash/skill.md)**

## Debugging

### Enable Logging

```bash
export FLASHINFER_JIT_VERBOSE=1      # Verbose JIT output
export FLASHINFER_JIT_DEBUG=1        # Debug symbols, -O0
export FLASHINFER_LOGLEVEL=3         # API logging (0=off, 1=basic, 3=detailed)
export FLASHINFER_LOGDEST=stdout
```

### Inspect Generated Code

```bash
# Generated sources
ls -la ~/.cache/flashinfer/0.6.0/*/generated/

# Compiled modules
ls -la ~/.cache/flashinfer/0.6.0/*/cached_ops/

# Build files
cat ~/.cache/flashinfer/0.6.0/*/cached_ops/*/build.ninja
```

### Environment Variables

```bash
# Compilation
export FLASHINFER_NVCC_THREADS=4              # Threads per nvcc process (--threads=N)
export MAX_JOBS=4                             # Parallel ninja jobs (nvcc processes)
export FLASHINFER_CUDA_ARCH_LIST="8.0 9.0a"  # Target architectures
# Memory note: total compilation memory ≈ MAX_JOBS × FLASHINFER_NVCC_THREADS × per-thread mem.

# Behavior
export FLASHINFER_WORKSPACE_BASE="/scratch"   # Custom cache directory
```

#### Full Environment Variable Reference

Already covered above (Quick Reference / Debugging): `FLASHINFER_LOGLEVEL`,
`FLASHINFER_LOGDEST`, `FLASHINFER_JIT_VERBOSE`, `FLASHINFER_JIT_DEBUG`,
`FLASHINFER_CUDA_ARCH_LIST`, `FLASHINFER_NVCC_THREADS`,
`FLASHINFER_WORKSPACE_BASE`, `MAX_JOBS`.

The remaining `FLASHINFER_*` knobs read by the code are listed below. Defaults
match what the code uses today; values are strings unless noted.

##### JIT / Build Toolchain

| Variable | Default | Read in | Effect |
|----------|---------|---------|--------|
| `FLASHINFER_DISABLE_JIT` | unset | `flashinfer/jit/core.py` | If set (any non-empty value), JIT compilation is refused and modules must already exist in the cache or be provided via AOT packages. |
| `FLASHINFER_CUTE_DSL_DISABLE_CACHE` | `0` | `flashinfer/jit/cute_dsl_core.py` | `1` disables the on-disk cache for JIT-compiled CuTe-DSL kernels (every process recompiles via `cute.compile`). |
| `FLASHINFER_DISABLE_VERSION_CHECK` | unset | `flashinfer/jit/env.py` | Skip the AOT/JIT-cache version check that pins flashinfer-jit-cache to the installed flashinfer-python. Bypass only when you intentionally mix versions. |
| `FLASHINFER_JIT_LINEINFO` | `0` | `flashinfer/jit/core.py` | `1` adds `-lineinfo` to nvcc so profiler / `cuda-gdb` can map PTX back to CUDA source. |
| `FLASHINFER_NVCC` | `$cuda_home/bin/nvcc` | `flashinfer/jit/cpp_ext.py` | Override the nvcc binary used by the JIT (useful for sccache wrappers or non-default CUDA installs). |
| `FLASHINFER_NVCC_LAUNCHER` | `""` | `flashinfer/jit/cpp_ext.py` | Optional launcher prefix for nvcc (e.g. `ccache`, `sccache`). Combined with `FLASHINFER_NVCC`. |
| `FLASHINFER_CXX_LAUNCHER` | `""` | `flashinfer/jit/cpp_ext.py` | Same idea as `FLASHINFER_NVCC_LAUNCHER` but for the host C++ compiler. |
| `FLASHINFER_FMHA_V2_VERBOSE` | unset | `flashinfer/jit/attention/fmha_v2/fmha_library.py` (FMHA v2 codegen, C++ side) | When set, the FMHA-v2 codegen / runtime prints verbose dispatcher diagnostics. Leave unset for normal runs. |
| `FLASHINFER_EXTRA_CFLAGS` | unset | `flashinfer/jit/cpp_ext.py` | Extra compiler flags passed to the host C++ compiler. |
| `FLASHINFER_EXTRA_CUDAFLAGS` | unset | `flashinfer/jit/cpp_ext.py` | Extra compiler flags passed to `nvcc`. |
| `FLASHINFER_EXTRA_LDFLAGS` | unset | `flashinfer/jit/cpp_ext.py` | Extra linker flags passed to the linker. |

##### Cubin / Artifact Loader

| Variable | Default | Read in | Effect |
|----------|---------|---------|--------|
| `FLASHINFER_CUBIN_DIR` | `<FLASHINFER_WORKSPACE_BASE>/.cache/flashinfer/cubins` | `flashinfer/jit/env.py` (exposed via `flashinfer/__main__.py show-config`) | Local directory used to cache downloaded cubins. Override to share a cache between users. |
| `FLASHINFER_CUBINS_REPOSITORY` | `https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local` | `flashinfer/jit/cubin_loader.py` | Base URL the loader downloads cubins from. Point to a mirror for offline or air-gapped setups. |
| `FLASHINFER_CUBIN_CHECKSUM_DISABLED` | unset | `flashinfer/jit/cubin_loader.py` | If set, skip SHA checksum verification of downloaded cubins. Debug aid only. |
| `FLASHINFER_CUBIN_DOWNLOAD_THREADS` | `4` | `flashinfer/artifacts.py` | Thread-pool size used by `flashinfer artifacts download`. |
| `FLASHINFER_NO_DOWNLOAD` | unset | `flashinfer/jit/cubin_loader.py` | Hard-fail if a cubin is missing locally instead of attempting to download. Useful in CI / locked-down environments. |
| `FLASHINFER_DSL_FMHA_LOCAL_DIR` | unset | `flashinfer/attention/cute_dsl/fmha.py` | Path to a local checkout of the CuTe-DSL FMHA kernel sources. The loader checks here before downloading. |
| `FLASHINFER_LOGGING_LEVEL` | `INFO` | `flashinfer/artifacts.py`, `flashinfer/jit/core.py` | Python logging level for the artifacts/cubin loader and the JIT compiler (`DEBUG`/`INFO`/`WARNING`/`ERROR`). Distinct from `FLASHINFER_LOGLEVEL`. |

##### API Dump / Logging Extensions

These complement `FLASHINFER_LOGLEVEL`/`FLASHINFER_LOGDEST` and are all read in `flashinfer/api_logging.py`.

| Variable | Default | Effect |
|----------|---------|--------|
| `FLASHINFER_DUMP_DIR` | `flashinfer_dumps` | Directory where `@flashinfer_api` writes per-call tensor dumps when logging is enabled. |
| `FLASHINFER_DUMP_INCLUDE` | `""` | Comma-separated allow-list of API names; only matching calls are dumped. Wildcards (`*`) supported. |
| `FLASHINFER_DUMP_EXCLUDE` | `""` | Comma-separated deny-list of API names; matching calls skip the (expensive) stats / dump path. |
| `FLASHINFER_DUMP_MAX_COUNT` | `1000` | Hard cap on number of dumped events; once reached, further dumps are dropped with a warning. |
| `FLASHINFER_DUMP_MAX_SIZE_GB` | `20` | Hard cap on total dump size in **gigabytes** (GB) written to `FLASHINFER_DUMP_DIR` (parsed as `float`; see `_DUMP_MAX_SIZE_GB` in `flashinfer/api_logging.py`). |
| `FLASHINFER_DUMP_SAFETENSORS` | `0` | `1` writes dumps as `.safetensors` (portable / Hugging Face-style); otherwise plain `.pt`. |

##### Trace Capture

Used by `flashinfer.trace` / `fi_trace`.

| Variable | Default | Read in | Effect |
|----------|---------|---------|--------|
| `FLASHINFER_TRACE_DUMP` | unset | `flashinfer/fi_trace.py` | If set, decorated APIs auto-dump benchmark-definition JSON for each call (the "fi_trace" feature). |
| `FLASHINFER_TRACE_DUMP_DIR` | cwd | `flashinfer/fi_trace.py` | Directory where the trace JSON files are written. |
| `FLASHINFER_TRACE_APPLY` | `0` | `flashinfer/__init__.py`, `flashinfer/trace_apply/config.py` | Set to `1` to enable Trace Apply (runtime kernel substitution) when FlashInfer is imported. |
| `FLASHINFER_TRACE_APPLY_PATH` | unset | `flashinfer/trace_apply/config.py` | Directory from which deployment-configured solutions are loaded for Trace Apply. |

##### Validation / Autotuning / Routing / Kernel Selection

| Variable | Default | Read in | Effect |
|----------|---------|---------|--------|
| `FLASHINFER_VALIDATE_INPUTS` | `0` | `flashinfer/mla/_core.py` (MLA wrapper) | Non-zero / non-empty value enables defensive input validation inside the MLA wrapper. Adds host-side overhead; intended for debugging. |
| `FLASHINFER_AUTOTUNER_LOAD_FROM_FILE` | `0` | `flashinfer/autotuner/autotuner.py` | `1` loads previously serialized autotune results from disk instead of re-running the search. |
| `FLASHINFER_AUTOTUNE_DIR` | unset | `flashinfer/mla/_sparse_mla_sm120.py` | Override the disk path for MLA AutoTuner cache files. Falls back to `FLASHINFER_WORKSPACE_DIR` when unset. |
| `FLASHINFER_AUTOTUNE_TIMER` | unset (auto) | `flashinfer/autotuner/autotuner.py` | Selects the autotuner's per-tactic timer: `globaltimer` forces the GPU `%globaltimer` register, `cuda_event` forces `cudaEvent`, unset/anything-else auto-detects (uses `%globaltimer` only when Confidential Computing is detected). Under CC `cudaEventElapsedTime` is unreliable (can go negative), so the globaltimer path keeps tactic ranking stable. |
| `FLASHINFER_CONFIDENTIAL_COMPUTE` | unset | `flashinfer/utils.py` | Override NVIDIA Confidential Computing (CC) auto-detection used by `is_confidential_compute()` (which drives the autotuner timer above): `1` forces CC, `0` forces non-CC. Useful for CI or hosts without `pynvml`. |
| `FLASHINFER_TOPK_ALGO` | unset | `flashinfer/topk.py` | Force a specific top-k algorithm (otherwise the dispatcher chooses based on shape). Used for benchmarking / regression bisection. |
| `FLASHINFER_USE_CUDA_NORM` | `0` | `flashinfer/norm/__init__.py` | `1` switches the norm path from the default backend to the legacy CUDA-only kernels. Diagnostic toggle. |
| `FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN` | unset | `csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu` | Forces the TRT-LLM MoE custom-routing kernel into "one-block-per-token" mode regardless of the active routing policy. Mainly used to reproduce specific perf points. |
| `FLASHINFER_B12X_MICRO_SHARE_INPUT` | `1` | `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py` | `0` disables the B12x MoE micro-batch input-sharing optimization. Internal/experimental — leave at the default unless investigating an SM12x MoE regression. |
| `FLASHINFER_B12X_FORCE_MOE_W4A16` | unset | `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py` | When set (any non-empty value), forces the SM12x MoE dispatcher onto the W4A16 kernel path regardless of weight dtype. Internal/experimental — used to reproduce W4A16-specific issues. |
| `FLASHINFER_TACTICS_BLOCKLIST` | unset | `flashinfer/autotuner/autotuner.py` | Path to a JSON tactics-blocklist file generated by `flashinfer tactics-blocklist generate` (or `python -m flashinfer tactics-blocklist generate`). When set, the autotuner loads the file at startup and skips any kernel tactics listed as invalid for the current GPU/driver environment, preventing hang or crash on known-bad tactics. |

## Development Workflow

### Typical Development Loop

1. Edit kernel code in `include/flashinfer/some_kernel.cuh`
2. Run test: `pytest tests/test_some_kernel.py::test_specific_case`
3. FlashInfer detects changes and recompiles automatically
4. No `pip install` needed!

### Modifying Existing Kernels

- **Kernel templates**: `include/flashinfer/**/*.cuh` - Changes picked up on next JIT compile
- **Launcher code**: `csrc/*.cu` - May need changes if adding new template parameters
- **Jinja templates**: `csrc/*.jinja` - Update if adding new config parameters
- **Python API**: `flashinfer/*.py` - Update if changing function signatures

### Creating Pre-compiled Packages

When ready to distribute:

```bash
# Build flashinfer-jit-cache package
cd flashinfer-jit-cache
export FLASHINFER_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a 11.0a 12.0f"
python -m build --no-isolation --wheel
```

This runs `flashinfer/aot.py` which calls all registered `gen_*_module()` functions and pre-compiles them.

## Build System Details

- **Build backend**: Custom PEP 517 backend in `build_backend.py`
- **Data directories**: Build creates symlinks for editable installs:
  - `3rdparty/cutlass` → `flashinfer/data/cutlass`
  - `csrc` → `flashinfer/data/csrc`
  - `include` → `flashinfer/data/include`
- **Version**: Generated in `flashinfer/_build_meta.py` from `version.txt`

## External Integrations

### TVM-FFI: Cross-Language Unified ABI

FlashInfer uses **TVM-FFI** (Apache TVM's Foreign Function Interface) for bindings, which provides a **cross-language unified ABI**. This means:

- **Not limited to PyTorch**: The same compiled kernels can be used from multiple frameworks
- **Language agnostic**: Bindings can be created for Python, C++, Rust, etc.
- **Type-safe marshaling**: Automatic tensor/array conversion between languages
- **Export syntax**: Use `TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, func)` to expose C++ functions

While FlashInfer currently provides PyTorch bindings, the underlying kernels are framework-agnostic thanks to TVM-FFI.

### Other Integrations

- **PyTorch Custom Ops**: `torch.library` for `torch.compile()` and CUDA graph support
- **Ninja Build**: Direct ninja generation, no CMake complexity

## Supported GPU Architectures

FlashInfer supports NVIDIA SM75, SM80, SM86, SM89, SM90, SM100, SM103, SM110, SM120, and SM121.

## Release Versioning

FlashInfer follows a "right-shifted" versioning scheme (`major.minor.patch[.post1]`):

- **major**: Architectural milestone and/or incompatible API changes (similar to PyTorch 2.0)
- **minor**: Significant backwards-compatible new features
- **patch**: Small backwards-compatible features (new kernels, new SM support) and backwards-compatible bug fixes
- **post1**: Optional suffix for quick follow-up release with just backwards-compatible bug fixes

## External Documentation Resources

When working with FlashInfer's dependencies and tools, refer to these official documentation sources:

### Core Dependencies

- **TVM-FFI**: Apache TVM's Foreign Function Interface
  - Documentation: <https://tvm.apache.org/ffi/>
  - Package: `apache-tvm-ffi` (<https://pypi.org/project/apache-tvm-ffi/>)
  - Use for: Understanding FFI export syntax, cross-language bindings

- **CUTLASS**: NVIDIA's CUDA Templates for Linear Algebra Subroutines
  - **Recommended**: Read source code directly in `3rdparty/cutlass/` (documentation is often outdated)
  - Repository: <https://github.com/NVIDIA/cutlass>
  - Use for: GEMM kernel implementations, tensor core operations

- **CuTe (CUTE DSL)**: CUTLASS's Cute Layout and Tensor DSL
  - Documentation: <https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl.html>
  - **Tip**: Add `.md` to get Markdown format: <https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl.html.md>
  - The Cute DSL kernels rely on Python modules from the `nvidia-cutlass-dsl` pip package, not to be confused with Python modules in the `3rdparty/cutlass` submodule
  - Tutorial: <https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL>

- **PTX ISA (Parallel Thread Execution)**: NVIDIA's PTX instruction set documentation
  - Documentation: <https://docs.nvidia.com/cuda/parallel-thread-execution/>
  - **Index/Table of Contents**: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.md>
  - **Tip**: Add `.md` to any page URL to get Markdown format
  - Use for: Low-level instruction details, new GPU architecture features, inline PTX assembly

### When to Consult These Docs

- **Understanding new GPU architecture features** → Check PTX ISA documentation for latest instruction details
- **Working on FFI bindings** → Check TVM-FFI docs for export patterns and type marshaling
- **Implementing Tensor-Core kernels using CUTLASS** → Read source code in `3rdparty/cutlass/`
- **Using tensor layouts or warp-level operations** → Refer to CuTe documentation
- **Writing inline PTX assembly** → Consult PTX ISA for instruction syntax and semantics

These dependencies are included in FlashInfer's `3rdparty/` directory or `requirements.txt`.

### Some final suggestions for all AI agents

> Because practical engineering involves the accumulated experience of trial and error, match the coding style, efficiency, complexity, verbosity, and defensiveness by learning from existing code as much as possible—this document contains many pointers on where to find examples. Document intentional departures with rationale. Mentioning "AI-assisted" in the git commit message is good transparency. For performance-critical hot paths, leave justification for the special algorithmic choices and other potential alternatives in a comment for review.

**Keep documentation in sync with code changes:** When modifying code that is referenced in this document or in `.claude/skills/`, update the corresponding documentation immediately. This includes:
- Important infrastructure changes (e.g., `@flashinfer_api`, `@backend_requirement`, TVM-FFI macros) → Update examples in `CLAUDE.md` and relevant skill files
- New patterns or conventions → Document them for future reference
- Deprecated approaches → Remove or mark as deprecated in docs
- New error handling patterns, macros, or utilities → Add to relevant skill tutorials
