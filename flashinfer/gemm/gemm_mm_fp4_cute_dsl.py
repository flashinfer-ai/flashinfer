"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Compile, disk-cache, and parallel-precompile helpers for the CuTe-DSL
block-scaled GEMM backend of mm_fp4.

Kept import-light on purpose: this module (not the much larger and more
frequently edited gemm_base.py) participates in the on-disk kernel cache
key, and parallel precompile workers import it in fresh subprocesses.
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_TORCH_TO_CUTLASS_DTYPE_ATTR = {
    torch.bfloat16: "BFloat16",
    torch.float16: "Float16",
}


def _blockscaled_gemm_cache_key_files() -> tuple:
    """Source files whose content invalidates the on-disk mm_fp4 kernels.

    Must be identical for every kernel of the module: the per-module
    ``meta.json`` records one source hash, so per-kernel variation would
    make sibling kernels wipe each other's artifacts.  This module (rather
    than the much larger, frequently edited gemm_base.py) is in the list
    because it constructs the ``cute.compile`` arguments.

    If the runner ever compiles other kernel classes into this cache
    module (e.g. re-enabling the SM103 kernel), their source files must
    be added here, or stale artifacts would survive kernel edits.
    """
    from .kernels import (
        dense_blockscaled_gemm_sm100,
        dense_blockscaled_gemm_sm100_common,
    )

    return (
        __file__,
        dense_blockscaled_gemm_sm100.__file__,
        dense_blockscaled_gemm_sm100_common.__file__,
    )


def _compile_block_scaled_gemm(
    cache,
    cache_key,
    make_gemm_kernel,
    ab_cutlass_dtype,
    sf_dtype,
    c_cutlass_dtype,
    ab_assumed_align,
    cluster_shape_mn,
    swap_ab,
    sf_m,
    sf_n,
    sf_k,
    batch_size,
    cluster_shape_k=1,
    cache_module_name=None,
):
    """Compile a block-scaled GEMM kernel via CuTe DSL and cache it.

    ``make_gemm_kernel`` is a zero-arg callable that returns a kernel instance
    (Sm100 or Sm103).  It is only invoked on a cache miss.

    When ``cache_module_name`` is given, compiled kernels also persist to the
    on-disk CuTe-DSL cache (see ``flashinfer/jit/cute_dsl_core.py``) and are
    reloaded from it in new processes; ``cache_key`` must therefore encode
    every codegen-affecting parameter.  ``max_active_clusters`` is baked into
    the persistent scheduler and is appended to the on-disk name, since the
    compile arch alone does not distinguish GPUs with different SM counts.

    TVM-FFI compilation pattern:
      - A, B, C, alpha: make_fake_compact_tensor -> torch tensors
        passed directly at runtime via TVM-FFI C-level dlpack
      - SF tensors: make_ptr (complex 6D BlockScaledBasicChunk
        layout can't be expressed as torch tensor) -> data_ptr() at runtime
      - Stream: make_fake_stream -> automatic env stream at runtime

    For FP4 runners, ``ab_cutlass_dtype`` is ``Uint8`` because FP4 data is
    stored as uint8 in torch (2 FP4 values per byte); the kernel wrapper
    recasts from Uint8 to Float4E2M1FN internally.
    """
    if cache_key in cache:
        return cache[cache_key]

    from flashinfer.cute_dsl.utils import get_max_active_clusters

    gemm = make_gemm_kernel()

    launch_cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1] * cluster_shape_k
    max_active_clusters = get_max_active_clusters(launch_cluster_size)

    compile_kernel = _make_blockscaled_gemm_compile_fn(
        gemm,
        ab_cutlass_dtype=ab_cutlass_dtype,
        sf_dtype=sf_dtype,
        c_cutlass_dtype=c_cutlass_dtype,
        ab_assumed_align=ab_assumed_align,
        swap_ab=swap_ab,
        sf_m=sf_m,
        sf_n=sf_n,
        sf_k=sf_k,
        batch_size=batch_size,
        max_active_clusters=max_active_clusters,
    )

    if cache_module_name is None:
        compiled_gemm = compile_kernel()
    else:
        from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel

        compiled_gemm = build_and_load_cute_dsl_kernel(
            cache_module_name,
            _blockscaled_kernel_disk_name(cache_key, batch_size, max_active_clusters),
            compile_kernel,
            extra_key_files=_blockscaled_gemm_cache_key_files(),
        )

    result = (compiled_gemm, max_active_clusters)
    cache[cache_key] = result
    return result


def _blockscaled_kernel_disk_name(cache_key, batch_size, max_active_clusters):
    """On-disk kernel name; shared by the in-process compile path and the
    parallel precompilation workers, which must agree byte-for-byte."""
    return "_".join(
        [str(part) for part in cache_key]
        + [f"b{batch_size}", f"mac{max_active_clusters}"]
    )


def _make_blockscaled_gemm_compile_fn(
    gemm,
    ab_cutlass_dtype,
    sf_dtype,
    c_cutlass_dtype,
    ab_assumed_align,
    swap_ab,
    sf_m,
    sf_n,
    sf_k,
    batch_size,
    max_active_clusters,
):
    """Zero-arg closure performing the ``cute.compile`` call for *gemm*."""
    import cutlass
    import cutlass.cute as cute

    from cutlass.cute.runtime import make_ptr

    def compile_kernel():
        sym_m = cute.sym_int()
        sym_k = cute.sym_int()
        sym_n = cute.sym_int()

        a_fake = cute.runtime.make_fake_compact_tensor(
            ab_cutlass_dtype,
            (sym_m, sym_k),
            stride_order=(1, 0),
            assumed_align=ab_assumed_align,
        )
        b_fake = cute.runtime.make_fake_compact_tensor(
            ab_cutlass_dtype,
            (sym_n, sym_k),
            stride_order=(1, 0),
            assumed_align=ab_assumed_align,
        )
        if swap_ab:
            c_fake = cute.runtime.make_fake_compact_tensor(
                c_cutlass_dtype,
                (sym_n, sym_m),
                stride_order=(0, 1),
                assumed_align=16,
            )
        else:
            c_fake = cute.runtime.make_fake_compact_tensor(
                c_cutlass_dtype,
                (sym_m, sym_n),
                stride_order=(1, 0),
                assumed_align=16,
            )

        a_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
        b_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
        alpha_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (1,), assumed_align=4
        )

        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        return cute.compile(
            gemm.wrapper,
            a_fake,
            b_fake,
            c_fake,
            sf_m,
            sf_n,
            sf_k,
            batch_size,
            a_sf_ptr,
            b_sf_ptr,
            alpha_fake,
            max_active_clusters,
            stream_fake,
            swap_ab,
            options="--opt-level 2 --enable-tvm-ffi",
        )

    return compile_kernel


def _mm_fp4_precompile_worker(payload):
    """Compile one mm_fp4 tactic in a spawned subprocess and persist it to
    the on-disk CuTe-DSL kernel cache.

    Returns ``(kernel_name, None)`` on success or ``(kernel_name, error)``;
    the parent logs failures and lets ``forward`` compile those tactics
    in-process on demand.
    """
    kernel_name = payload["kernel_name"]
    try:
        import cutlass

        from ..jit.cute_dsl_core import JitSpecCuteDsl, _hash_source_files
        from .kernels.dense_blockscaled_gemm_sm100 import (
            Sm100BlockScaledPersistentDenseGemmKernel,
        )

        (
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            swap_ab,
            use_prefetch,
            _kernel_type,
            _use_tma_store,
            enable_pdl,
            out_dtype,
        ) = payload["cache_key"]

        gemm = Sm100BlockScaledPersistentDenseGemmKernel(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
        )
        compile_fn = _make_blockscaled_gemm_compile_fn(
            gemm,
            ab_cutlass_dtype=cutlass.Uint8,
            sf_dtype=cutlass.Float8E4M3FN
            if sf_vec_size == 16
            else cutlass.Float8E8M0FNU,
            c_cutlass_dtype=getattr(cutlass, _TORCH_TO_CUTLASS_DTYPE_ATTR[out_dtype]),
            ab_assumed_align=32,
            swap_ab=swap_ab,
            sf_m=payload["sf_m"],
            sf_n=payload["sf_n"],
            sf_k=payload["sf_k"],
            batch_size=payload["batch_size"],
            max_active_clusters=payload["max_active_clusters"],
        )
        spec = JitSpecCuteDsl(
            "mm_fp4",
            kernel_name,
            compile_fn,
            _hash_source_files(tuple(payload["key_files"])),
        )
        if not spec.is_compiled:
            spec.compile_and_persist()
        return (kernel_name, None)
    except Exception as e:  # noqa: BLE001 -- reported to the parent
        return (kernel_name, f"{type(e).__name__}: {e}")


# Host-RAM budget per precompile worker.  Measured peak RSS per worker is
# ~0.8 GB (torch/cutlass import baseline + one cute.compile); 1 GB adds
# headroom.  cute.compile itself uses zero GPU memory.
_MM_FP4_PRECOMPILE_WORKER_RAM_BYTES = 1 << 30


def _available_host_memory_bytes() -> Optional[int]:
    """Return MemAvailable from /proc/meminfo, or None if unreadable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _get_mm_fp4_cute_dsl_compile_workers() -> int:
    """Number of subprocesses used to precompile mm_fp4 cute-dsl tactics
    during autotuning.

    Override with FLASHINFER_MM_FP4_CUTE_DSL_COMPILE_WORKERS; <= 1 disables
    parallel precompilation (tactics compile serially on first use, as
    before).  The requested count is additionally capped by available host
    memory to avoid host-side OOM when many processes tune concurrently.
    """
    workers = int(os.environ.get("FLASHINFER_MM_FP4_CUTE_DSL_COMPILE_WORKERS", "4"))
    if workers <= 1:
        return workers
    available = _available_host_memory_bytes()
    if available is not None:
        mem_cap = int(available // _MM_FP4_PRECOMPILE_WORKER_RAM_BYTES)
        if mem_cap < workers:
            logger.warning(
                f"[mm_fp4 cute-dsl] capping tactic precompile workers "
                f"{workers} -> {mem_cap} (host MemAvailable = "
                f"{available / (1 << 30):.1f} GiB)."
            )
            workers = mem_cap
    return workers


def run_mm_fp4_precompile_pool(payloads) -> None:
    """Compile mm_fp4 tactics into the on-disk cache with a subprocess pool.

    Each payload is handled by ``_mm_fp4_precompile_worker``; failures are
    logged and skipped (those tactics compile in-process on first use).
    """
    from multiprocessing import get_context

    num_workers = min(_get_mm_fp4_cute_dsl_compile_workers(), len(payloads))
    logger.info(
        f"[mm_fp4 cute-dsl] precompiling {len(payloads)} tactics "
        f"with {num_workers} workers"
    )
    with get_context("spawn").Pool(num_workers) as pool:
        results = pool.map(_mm_fp4_precompile_worker, payloads)
    for kernel_name, err in results:
        if err is not None:
            logger.debug(
                f"[mm_fp4 cute-dsl] precompile failed for {kernel_name}: {err}"
            )
