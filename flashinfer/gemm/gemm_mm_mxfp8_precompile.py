"""Precompile SM100-family MXFP8 tactics into the CuTe-DSL disk cache."""

import hashlib
import logging
import os
from multiprocessing import get_context

import torch
from filelock import FileLock

from ..utils import get_device_index
from .gemm_mm_fp4_cute_dsl import (
    _available_host_memory_bytes,
    _blockscaled_mxfp8_gemm_cache_key_files,
    _make_blockscaled_gemm_compile_fn,
    _mm_mxfp8_cache_key,
    _mxfp8_blockscaled_kernel_disk_name,
)

logger = logging.getLogger(__name__)


def _get_mm_mxfp8_compile_workers():
    workers = int(os.environ.get("FLASHINFER_MM_MXFP8_CUTE_DSL_COMPILE_WORKERS", "4"))
    if workers <= 1:
        return workers
    cpu_count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    available = _available_host_memory_bytes()
    # Match the FP4 compiler's conservative per-worker RAM allowance.
    memory_cap = workers if available is None else available // (1 << 30)
    return min(workers, cpu_count, memory_cap)


def _mm_mxfp8_precompile_worker(payload):
    name = payload["kernel_name"]
    try:
        torch.cuda.set_device(payload["device_index"])
        import cutlass

        from ..cute_dsl.utils import torch_to_cutlass_dtype
        from ..jit.cute_dsl_core import JitSpecCuteDsl
        from .kernels.dense_blockscaled_gemm_sm100 import (
            Sm100BlockScaledPersistentDenseGemmKernel,
        )
        from .kernels.dense_blockscaled_gemm_sm100_splitk import (
            Sm100BlockScaledSplitKGemmKernel,
        )

        sf_vec_size, tile, cluster, swap, prefetch, pdl, dtype, slices = payload[
            "cache_key"
        ]
        kernel = (
            Sm100BlockScaledSplitKGemmKernel(sf_vec_size, tile, slices, pdl)
            if slices > 1
            else Sm100BlockScaledPersistentDenseGemmKernel(
                sf_vec_size, tile, cluster, prefetch, pdl
            )
        )
        compile_fn = _make_blockscaled_gemm_compile_fn(
            kernel,
            ab_cutlass_dtype=cutlass.Float8E4M3FN,
            sf_dtype=cutlass.Float8E8M0FNU,
            c_cutlass_dtype=torch_to_cutlass_dtype(dtype),
            ab_assumed_align=16,
            swap_ab=swap,
            sf_m=payload["sf_m"],
            sf_n=payload["sf_n"],
            sf_k=payload["sf_k"],
            batch_size=1,
            max_active_clusters=payload["max_active_clusters"],
        )
        spec = JitSpecCuteDsl("mm_mxfp8", name, compile_fn, payload["source_sha256"])
        if not spec.is_compiled:
            spec.compile_and_persist()
        return name, None
    except Exception as exc:  # noqa: BLE001 -- forward retries missing artifacts
        return name, f"{type(exc).__name__}: {exc}"


def _run_mm_mxfp8_precompile_pool(payloads, workers):
    logger.info(
        "[mm_mxfp8 cute-dsl] precompiling %d tactics with %d workers",
        len(payloads),
        workers,
    )
    with get_context("spawn").Pool(workers) as pool:
        results = pool.map(_mm_mxfp8_precompile_worker, payloads)
    for name, error in results:
        if error is not None:
            logger.warning(
                "[mm_mxfp8 cute-dsl] precompile failed for %s: %s; retrying on demand",
                name,
                error,
            )


def precompile_mm_mxfp8_tactics(
    tactics, m, n, k, enable_pdl, out_dtype, kernel_cache, device
):
    from ..cute_dsl.utils import get_max_active_clusters
    from ..jit.core import get_tmpdir
    from ..jit.cute_dsl_core import (
        JitSpecCuteDsl,
        _hash_source_files,
        cute_dsl_cache_disabled,
    )

    if (
        os.environ.get("FLASHINFER_DISABLE_JIT")
        or cute_dsl_cache_disabled()
        or _get_mm_mxfp8_compile_workers() <= 1
    ):
        return
    device_index = get_device_index(device)
    # Occupancy and compilation-target lookup both depend on the current device.
    with torch.cuda.device(device_index):
        source_hash = _hash_source_files(_blockscaled_mxfp8_gemm_cache_key_files())
        module = JitSpecCuteDsl("mm_mxfp8", "precompile", None, source_hash)
        cache_identity = hashlib.sha256(str(module.module_dir).encode()).hexdigest()[
            :16
        ]
        # TP ranks share one pool at a time on a host/cache. Recheck disk entries
        # after acquiring the lock, so waiting ranks do not compile them again.
        with FileLock(
            get_tmpdir() / f"mm_mxfp8_{cache_identity}.precompile.lock",
            thread_local=False,
        ):
            workers = _get_mm_mxfp8_compile_workers()
            if workers <= 1:
                return
            max_clusters = {}
            payloads = {}
            for tile, cluster, swap, prefetch, slices in tactics:
                key = _mm_mxfp8_cache_key(
                    32, tile, cluster, swap, prefetch, enable_pdl, out_dtype, slices
                )
                if (device_index, key) in kernel_cache:
                    continue
                cluster_size = cluster[0] * cluster[1] * slices
                if cluster_size not in max_clusters:
                    max_clusters[cluster_size] = get_max_active_clusters(cluster_size)
                mac = max_clusters[cluster_size]
                name = _mxfp8_blockscaled_kernel_disk_name(key, 1, mac)
                spec = JitSpecCuteDsl("mm_mxfp8", name, None, source_hash)
                if spec.is_compiled:
                    continue
                kernel_m, kernel_n = (n, m) if swap else (m, n)
                payloads[name] = dict(
                    kernel_name=name,
                    cache_key=key,
                    device_index=device_index,
                    sf_m=(kernel_m + 127) // 128,
                    sf_n=(kernel_n + 127) // 128,
                    sf_k=(k // 32 + 3) // 4,
                    max_active_clusters=mac,
                    source_sha256=source_hash,
                )
            # One new specialization is cheaper to compile in the caller.
            if len(payloads) > 1:
                _run_mm_mxfp8_precompile_pool(
                    list(payloads.values()), min(workers, len(payloads))
                )
