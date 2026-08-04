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
"""

import logging
import os
from typing import Optional

import torch

from ..utils import get_device_index

logger = logging.getLogger(__name__)


def _blockscaled_gemm_cache_key_files() -> tuple:
    """Source files whose content invalidates the on-disk mm_fp4 kernels."""
    from .kernels import (
        dense_blockscaled_gemm_sm100,
        dense_blockscaled_gemm_sm100_common,
        dense_blockscaled_gemm_sm103,
    )

    return (
        __file__,
        dense_blockscaled_gemm_sm100.__file__,
        dense_blockscaled_gemm_sm100_common.__file__,
        dense_blockscaled_gemm_sm103.__file__,
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
    device_index=None,
):
    """Compile a block-scaled GEMM kernel via CuTe DSL and cache it.

    ``make_gemm_kernel`` is a zero-arg callable that returns a kernel instance
    (Sm100 or Sm103).  It is only invoked on a cache miss.

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
    if device_index is None:
        device_index = torch.cuda.current_device()
    mem_key = (device_index, cache_key)
    if mem_key in cache:
        return cache[mem_key]

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
    cache[mem_key] = result
    return result


def _blockscaled_kernel_disk_name(cache_key, batch_size, max_active_clusters):
    """On-disk kernel name encoding every mm_fp4 codegen parameter.

    Must be symbol-safe as produced (see tests/jit/test_cute_dsl_cache.py):
    JitSpecCuteDsl sanitizes names, and two names differing only in
    sanitized-away characters would collide on one artifact.
    """
    (
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        swap_ab,
        use_prefetch,
        kernel_type,
        use_tma_store,
        enable_pdl,
        out_dtype,
    ) = cache_key
    tma = "x" if use_tma_store is None else int(use_tma_store)
    dtype = str(out_dtype).removeprefix("torch.")
    return (
        f"sf{sf_vec_size}_t{mma_tiler_mn[0]}x{mma_tiler_mn[1]}"
        f"_c{cluster_shape_mn[0]}x{cluster_shape_mn[1]}"
        f"_swap{int(swap_ab)}_pf{int(use_prefetch)}_{kernel_type}"
        f"_tma{tma}_pdl{int(enable_pdl)}_{dtype}"
        f"_b{batch_size}_mac{max_active_clusters}"
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
    """Build a zero-arg closure that runs ``cute.compile`` for gemm."""
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
        # Set the current GPU in this subprocess
        torch.cuda.set_device(payload["device_index"])

        import cutlass

        from ..cute_dsl.utils import torch_to_cutlass_dtype
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
            c_cutlass_dtype=torch_to_cutlass_dtype(out_dtype),
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


# Empirically measured host-RAM budget (RSS) per precompile worker -- 1 GiB
_MM_FP4_PRECOMPILE_WORKER_RAM_BYTES = 1 << 30


def _cgroup_available_memory_bytes() -> Optional[int]:
    """Headroom under the current cgroup memory limit, or None if
    unlimited/unreadable.

    Inside containers /proc/meminfo reports *host* memory, so the cgroup
    limit is what actually prevents an OOM kill.
    cgroup's memory.current is profiled here to estimate the headroom.
    """
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            limit = f.read().strip()
        if limit != "max":
            with open("/sys/fs/cgroup/memory.current") as f:
                current = int(f.read())
            return max(0, int(limit) - current)
    except (OSError, ValueError):
        pass
    # cgroup v1
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            limit_v1 = int(f.read())
        if limit_v1 < 1 << 60:  # values near 2**63 mean "unlimited"
            with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
                usage = int(f.read())
            return max(0, limit_v1 - usage)
    except (OSError, ValueError):
        pass
    return None


def _available_host_memory_bytes() -> Optional[int]:
    """Best-effort available memory: the tighter of host MemAvailable and
    the cgroup limit headroom, or None if neither is readable."""
    candidates = []
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    candidates.append(int(line.split()[1]) * 1024)
                    break
    except OSError:
        pass
    cgroup = _cgroup_available_memory_bytes()
    if cgroup is not None:
        candidates.append(cgroup)
    return min(candidates) if candidates else None


def _get_mm_fp4_cute_dsl_compile_workers() -> int:
    """How many subprocesses to use for precompiling mm_fp4 cute-dsl tactics.

    Starts from FLASHINFER_MM_FP4_CUTE_DSL_COMPILE_WORKERS (default 4), then
    lowers it to what host RAM can hold using 1 GiB per spawned worker as a safety measure.
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


def _run_mm_fp4_precompile_pool(payloads) -> None:
    """Compile mm_fp4 tactics into the on-disk cache with a subprocess pool.

    Each subprocess runs _mm_fp4_precompile_worker.
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


def _mm_fp4_cache_key(sf_vec_size, tactic, enable_pdl, out_dtype):
    """In-memory kernel-cache key for one mm_fp4 tactic tuple.

    Shared by the runner's forward path and the precompile path, which
    must agree byte-for-byte: the on-disk kernel name derives from it.
    """
    return (sf_vec_size, *tactic, enable_pdl, out_dtype)


def precompile_mm_fp4_tactics(
    tactics, m, n, real_k, use_nvfp4, enable_pdl, out_dtype, kernel_cache, device
) -> None:
    """Batch-compile not-yet-cached mm_fp4 tactics into the on-disk
    CuTe-DSL cache with a pool of subprocesses.

    Called by autotuner's do_preparation, before the per-tactic profiling loop.
    Failures are non-fatal: any tactic missing from kernel_cache and
    the disk cache compiles in-process on first use.
    """
    from ..jit.cute_dsl_core import (
        JitSpecCuteDsl,
        _hash_source_files,
        cute_dsl_cache_disabled,
    )

    if cute_dsl_cache_disabled():
        return  # workers hand kernels to the parent via the disk cache
    if _get_mm_fp4_cute_dsl_compile_workers() <= 1:
        return

    from flashinfer.cute_dsl.utils import get_max_active_clusters

    sf_vec_size = 16 if use_nvfp4 else 32
    device_index = get_device_index(device)
    key_files = _blockscaled_gemm_cache_key_files()
    source_sha256 = _hash_source_files(tuple(key_files))

    max_clusters_cache: dict = {}
    payloads = []
    for tactic in tactics:
        (
            mma_tiler_mn,
            cluster_shape_mn,
            swap_ab,
            _use_prefetch,
            kernel_type,
            _use_tma_store,
        ) = tactic
        if kernel_type != "sm100":
            continue
        cache_key = _mm_fp4_cache_key(sf_vec_size, tactic, enable_pdl, out_dtype)
        if (device_index, cache_key) in kernel_cache:
            continue

        cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
        if cluster_size not in max_clusters_cache:
            max_clusters_cache[cluster_size] = get_max_active_clusters(cluster_size)
        mac = max_clusters_cache[cluster_size]
        kernel_name = _blockscaled_kernel_disk_name(cache_key, 1, mac)
        spec = JitSpecCuteDsl("mm_fp4", kernel_name, lambda: None, source_sha256)
        if spec.is_compiled:
            continue  # already on disk; forward will JITLink it

        kernel_m, kernel_n = (n, m) if swap_ab else (m, n)
        payloads.append(
            {
                "cache_key": cache_key,
                "kernel_name": kernel_name,
                "max_active_clusters": mac,
                "sf_m": (kernel_m + 127) // 128,
                "sf_n": (kernel_n + 127) // 128,
                "sf_k": (real_k // sf_vec_size + 3) // 4,
                "batch_size": 1,
                "key_files": key_files,
                "device_index": device_index,
            }
        )

    # A single missing tactic compiles faster in-process than the
    # spawn + import cost of a one-worker pool.
    if len(payloads) < 2:
        return

    _run_mm_fp4_precompile_pool(payloads)
