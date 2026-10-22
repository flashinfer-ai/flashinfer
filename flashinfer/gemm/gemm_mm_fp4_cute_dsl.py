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

import functools
import logging
import os

import torch

from ..utils import get_device_index
from .gemm_blockscaled_cute_dsl import _get_sm100_block_scaled_tactics
from .kernels.utils import _CLUSTER_SHAPE_MN_CANDIDATES

logger = logging.getLogger(__name__)


# kernel_type of the low-M cluster split-K tactics of mm_fp4(backend="cute-dsl");
# their use_tma_store slot carries the K-slice count (2 or 4).
_SM100_SPLITK_KERNEL_TYPE = "sm100sk"
# Deep-K persistent tactic: 8 MMA K instructions per stage (K tile 512 for FP4)
# instead of 4, so every TMA row fetch is 256 B. Carried in the use_tma_store
# slot of an "sm100" tactic. Wins for narrow (<= 32) token tiles once the
# weight grid is about a wave or more (>= _SM100_DEEP_K_MIN_TILES weight
# tiles); below that the longer pipeline fill/drain costs 1-2 %.
_SM100_DEEP_K_INST = 8


_SM100_DEEP_K_TILE = 512


@functools.cache
def _get_mm_fp4_kernel_classes(sm_version: int, require_native: bool = False):
    """Resolve optional FP4 families without making runtime/AOT callers import them.

    SM103 currently uses the SM100 family. Native Rubin and 3xFP4 families
    are offered on SM107 when their DSL APIs are present; AOT requires both.
    """
    sm103 = sm107 = None
    if sm_version == 107:
        try:
            from .kernels.dense_blockscaled_gemm_sm103 import (
                Sm103BlockScaledPersistentDenseGemmKernel,
            )

            sm103 = Sm103BlockScaledPersistentDenseGemmKernel
        except ImportError:
            if require_native:
                raise
        try:
            from .kernels.dense_blockscaled_gemm_sm107 import (
                Sm107BlockScaledPersistentDenseGemmKernel,
            )

            sm107 = Sm107BlockScaledPersistentDenseGemmKernel
        except ImportError:
            if require_native:
                raise
    return sm103, sm107


def _get_mm_fp4_cute_dsl_tactics(
    m,
    n,
    real_k,
    use_nvfp4,
    c_cutlass_dtype,
    device,
    sm_version,
    out_contiguous=True,
    per_token_alpha=False,
):
    """Unranked tactic space shared by runtime selection and CPU AOT builds."""
    import cutlass
    from .kernels.dense_blockscaled_gemm_sm100_splitk import (
        Sm100BlockScaledSplitKGemmKernel as _SplitKKernel,
    )

    Sm103Kernel, Sm107Kernel = _get_mm_fp4_kernel_classes(sm_version)

    sf_vec_size = 16 if use_nvfp4 else 32
    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN if use_nvfp4 else cutlass.Float8E8M0FNU

    # SM100 tactics (shared enumeration with mxfp8)
    sm100_base = _get_sm100_block_scaled_tactics(
        m,
        n,
        real_k,
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_cutlass_dtype,
        device,
    )

    if m <= 32:
        allowed_tiles = {
            (128, 8),
            (128, 16),
            (128, 32),
            (256, 8),
            (256, 16),
            (256, 32),
        }

        sm100_base = [t for t in sm100_base if t[0] in allowed_tiles and t[2]]

    valid_tactics = [(*t, "sm100", None) for t in sm100_base]
    # Deep-K variant (K tile 512, use_tma_store slot = 8 MMA K
    # instructions per stage) for narrow N tiles; see
    # _select_sm100_mm_fp4_splitk_tactic for where it wins untuned.
    if real_k % _SM100_DEEP_K_TILE == 0:
        valid_tactics += [
            (*t, "sm100", _SM100_DEEP_K_INST) for t in sm100_base if t[0][1] <= 32
        ]

    # Low-M cluster split-K (swap_ab only; the use_tma_store slot
    # carries the K-slice count). Its epilogue applies the per-token
    # alpha after the FP32 cluster reduction, so it stays valid for
    # per-token alpha.
    if use_nvfp4 and out_contiguous and _SplitKKernel.supports_m(m):
        for split_k_slices in _SplitKKernel.SUPPORTED_SPLIT_K_SLICES:
            if not _SplitKKernel.is_valid_tactic(m, real_k, ab_dtype, split_k_slices):
                continue
            for sk_tile in _SplitKKernel.mma_tilers_for_m(m):
                valid_tactics.append(
                    (
                        sk_tile,
                        (1, 1),
                        True,
                        False,
                        _SM100_SPLITK_KERNEL_TYPE,
                        split_k_slices,
                    )
                )

    # Shared by the SM103 and SM107 tactic blocks below. Hoisted out of
    # the SM103 block: the two blocks have independent guards (the SM103
    # kernel needs the internal cutlass-dsl wheel, the SM107 one needs
    # rubin_helpers), so on a public DSL with Sm103Kernel None but
    # Sm107Kernel present the SM107 block would otherwise reference these
    # before assignment.
    batch_size = 1
    m_aligned = m % 8 == 0
    n_aligned = n % 8 == 0

    # --- SM103 tactics (only on SM103) ---
    if sm_version in [103, 107] and Sm103Kernel is not None:
        sm103_mma_tiler_candidates = [
            (128, 128),
            (256, 128),
            (128, 256),
            (256, 256),
        ]

        for mma_tiler_mn in sm103_mma_tiler_candidates:
            for cluster_shape_mn in _CLUSTER_SHAPE_MN_CANDIDATES:
                for swap_ab in (False, True):
                    if not swap_ab and not n_aligned:
                        continue
                    if swap_ab and not m_aligned:
                        continue

                    if swap_ab:
                        c_major = "m"
                        kernel_m, kernel_n = n, m
                    else:
                        c_major = "n"
                        kernel_m, kernel_n = m, n

                    for use_tma_store in (True, False):
                        if not Sm103Kernel.can_implement(
                            ab_dtype,
                            sf_dtype,
                            sf_vec_size,
                            c_cutlass_dtype,
                            mma_tiler_mn,
                            cluster_shape_mn,
                            kernel_m,
                            kernel_n,
                            real_k,
                            batch_size,
                            "k",
                            "k",
                            c_major,
                            use_tma_store,
                        ):
                            continue

                        valid_tactics.append(
                            (  # type: ignore[arg-type]
                                mma_tiler_mn,
                                cluster_shape_mn,
                                swap_ab,
                                False,
                                "sm103",
                                use_tma_store,
                            )
                        )

    # --- SM107 tactics (only on SM107) ---
    if sm_version == 107 and Sm107Kernel is not None:
        sm107_mma_tiler_mn_candidates = [
            (128, 64),
            (256, 64),
            (128, 128),
            (256, 128),
            (128, 192),
            (256, 192),
            (128, 256),
            (256, 256),
        ]
        sm107_mma_inst_shape_m_candidates = [128, 256]

        for mma_tiler_mn in sm107_mma_tiler_mn_candidates:
            for mma_inst_shape_m in sm107_mma_inst_shape_m_candidates:
                mma_inst_shape_k = 128  # fixed for FP4
                mma_tiler_k = 256  # fixed for FP4
                mma_inst_shape = (
                    mma_inst_shape_m,
                    mma_tiler_mn[1],
                    mma_inst_shape_k,
                )

                for cluster_shape_mn in _CLUSTER_SHAPE_MN_CANDIDATES:
                    for swap_ab in (False, True):
                        if not swap_ab and not n_aligned:
                            continue
                        if swap_ab and not m_aligned:
                            continue

                        if swap_ab:
                            c_major = "m"
                            kernel_m, kernel_n = n, m
                        else:
                            c_major = "n"
                            kernel_m, kernel_n = m, n

                        if not Sm107Kernel.can_implement(
                            ab_dtype,
                            sf_dtype,
                            sf_vec_size,
                            c_cutlass_dtype,
                            mma_tiler_mn,
                            mma_inst_shape,
                            cluster_shape_mn,
                            kernel_m,
                            kernel_n,
                            real_k,
                            batch_size,
                            "k",
                            "k",
                            c_major,
                        ):
                            continue

                        # prefetch_dist: 0=off (best default at generic
                        # shapes), 2=shallow, None=auto (num_ab_stage
                        # depth; helps at long-K DSV4-like shapes)
                        for prefetch_dist in (0, 2, None):
                            valid_tactics.append(
                                (  # type: ignore[arg-type]
                                    mma_tiler_mn,
                                    cluster_shape_mn,
                                    swap_ab,
                                    False,  # use_prefetch is SM100-only
                                    "sm107",
                                    (
                                        mma_inst_shape_m,
                                        mma_tiler_mn[1],
                                        mma_inst_shape_k,
                                        mma_tiler_k,
                                        prefetch_dist,
                                    ),
                                )
                            )

    if per_token_alpha:
        # Only the SM100 kernel has the per-row alpha epilogue. The
        # SM103/SM107 tactics would each raise in forward() and cost the
        # tuner a profiling pass for a tactic that can never win.
        valid_tactics = [
            t for t in valid_tactics if t[4] in ("sm100", _SM100_SPLITK_KERNEL_TYPE)
        ]

    return valid_tactics


def _blockscaled_gemm_cache_key_files() -> tuple:
    """Source files whose content invalidates the on-disk mm_fp4 kernels."""
    from .kernels import (
        dense_blockscaled_gemm_sm100,
        dense_blockscaled_gemm_sm100_common,
        dense_blockscaled_gemm_sm103,
        dense_blockscaled_gemm_sm100_splitk,
        epilogue_utils,
    )

    return (
        __file__,
        dense_blockscaled_gemm_sm100.__file__,
        dense_blockscaled_gemm_sm100_common.__file__,
        dense_blockscaled_gemm_sm103.__file__,
        dense_blockscaled_gemm_sm100_splitk.__file__,
        epilogue_utils.__file__,
        # Hash Rubin sources without importing its optional DSL dependencies.
        os.path.join(
            os.path.dirname(__file__), "kernels", "dense_blockscaled_gemm_sm107.py"
        ),
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
    per_token_alpha=None,
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
        per_token_alpha=per_token_alpha,
        dynamic_max_active_clusters=cache_module_name == "mm_fp4",
    )

    if cache_module_name is None:
        compiled_gemm = compile_kernel()
    else:
        from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel

        compiled_gemm = build_and_load_cute_dsl_kernel(
            cache_module_name,
            _blockscaled_kernel_disk_name(
                cache_key,
                batch_size,
                None if cache_module_name == "mm_fp4" else max_active_clusters,
            ),
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
        per_token_alpha,
        l2_policy,
    ) = cache_key
    # On SM107 the use_tma_store slot is repurposed to carry the Rubin kernel
    # shape (inst_m, inst_n, inst_k, tiler_k, prefetch_dist), so render it as a
    # symbol-safe joined string; a bare int() would raise on the tuple.
    if use_tma_store is None:
        tma = "x"
    elif isinstance(use_tma_store, tuple):
        # prefetch_dist is enumerated as (0, 2, None) - None means "auto" and
        # compiles differently from 0, so render it as its own symbol rather
        # than letting int(None) raise.
        tma = "s" + "s".join("x" if v is None else str(int(v)) for v in use_tma_store)
    else:
        tma = int(use_tma_store)
    dtype = str(out_dtype).removeprefix("torch.")
    alpha = "x" if per_token_alpha is None else per_token_alpha
    l2 = "x" if l2_policy is None else l2_policy
    return (
        f"sf{sf_vec_size}_t{mma_tiler_mn[0]}x{mma_tiler_mn[1]}"
        f"_c{cluster_shape_mn[0]}x{cluster_shape_mn[1]}"
        f"_swap{int(swap_ab)}_pf{int(use_prefetch)}_{kernel_type}"
        f"_tma{tma}_pdl{int(enable_pdl)}_{dtype}"
        f"_pta{alpha}_l2{l2}"
        f"_b{batch_size}_mac{max_active_clusters if max_active_clusters is not None else 'dynamic'}"
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
    per_token_alpha=None,
    dynamic_max_active_clusters=False,
):
    """Build a zero-arg closure that runs ``cute.compile`` for gemm."""
    import cutlass
    import cutlass.cute as cute

    from cutlass.cute.runtime import make_ptr

    # Keep the shared GEMM wrapper's existing constexpr ABI for other callers
    # (MXFP8 etc.). FP4 uses a runtime occupancy argument so one exported object
    # works across GPU SKUs and can be compiled on a CPU release builder.
    @cute.jit
    def fp4_wrapper(
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        l: cutlass.Constexpr,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        max_active_clusters: cutlass.Int32,
        current_stream,
        swap_ab: cutlass.Constexpr,
    ):
        gemm.wrapper(
            mA,
            mB,
            mC,
            sf_m,
            sf_n,
            sf_k,
            l,
            a_sf_ptr,
            b_sf_ptr,
            alpha_tensor,
            max_active_clusters,
            current_stream,
            swap_ab,
        )

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
        # Binding alpha to the symbolic extent the tokens live on keeps one
        # compiled kernel across all token counts.
        alpha_extent = {None: 1, "m": sym_m, "n": sym_n}[per_token_alpha]
        alpha_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (alpha_extent,), assumed_align=4
        )

        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        return cute.compile(
            fp4_wrapper if dynamic_max_active_clusters else gemm.wrapper,
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
            cutlass.Int32(max_active_clusters)
            if dynamic_max_active_clusters
            else max_active_clusters,
            stream_fake,
            swap_ab,
            options="--opt-level 2 --enable-tvm-ffi",
        )

    return compile_kernel


_CUTE_DSL_ALPHA_ONE_CACHE: dict = {}


def _prepare_alpha_for_launch(alpha_tensor, device):
    """Prepare alpha as a 1-dim float32 device tensor with shape [1].

    When *alpha_tensor* is ``None``, returns a cached ``tensor([1.0])``
    on *device* (allocated once, reused forever).
    """
    if alpha_tensor is None:
        cached = _CUTE_DSL_ALPHA_ONE_CACHE.get(device)
        if cached is None:
            cached = torch.tensor([1.0], dtype=torch.float32, device=device)
            _CUTE_DSL_ALPHA_ONE_CACHE[device] = cached
        return cached
    if alpha_tensor.dim() == 0:
        return alpha_tensor.unsqueeze(0)
    return alpha_tensor.reshape(1)


def _mm_fp4_precompile_worker(payload):
    """Compile one mm_fp4 tactic in a spawned subprocess and persist it to
    the on-disk CuTe-DSL kernel cache.

    Returns ``(kernel_name, None)`` on success or ``(kernel_name, error)``;
    the parent logs failures and lets ``forward`` compile those tactics
    in-process on demand.
    """
    kernel_name = payload["kernel_name"]
    try:
        torch.cuda.set_device(payload["device_index"])

        import cutlass

        from ..cute_dsl.utils import torch_to_cutlass_dtype
        from ..jit.cute_dsl_core import JitSpecCuteDsl, _hash_source_files

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
            per_token_alpha,
            l2_policy,
        ) = payload["cache_key"]

        # The use_tma_store slot of an "sm100" tactic carries the MMA K
        # instructions per stage (None -> 4, 8 -> K tile 512); the worker must
        # build the same kernel the runner would, since the disk name derives
        # from the cache key.
        gemm = _make_mm_fp4_kernel(payload["cache_key"])
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
            per_token_alpha=per_token_alpha,
            dynamic_max_active_clusters=True,
        )
        spec = JitSpecCuteDsl(
            "mm_fp4",
            kernel_name,
            compile_fn,
            _hash_source_files(tuple(payload["key_files"])),
        )
        if not spec.is_compiled:
            spec.compile_and_persist()
        # Report failed exports so the parent can fall back to serial compilation.
        if not spec.is_compiled:
            raise RuntimeError(
                f"CuTe-DSL artifact was not persisted: {spec.object_path}"
            )
        return (kernel_name, None)
    except Exception as e:  # noqa: BLE001 -- reported to the parent
        return (kernel_name, f"{type(e).__name__}: {e}")


def _make_mm_fp4_kernel(cache_key):
    """Construct the same kernel for a runtime tactic or an AOT worker."""
    (
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        swap_ab,
        use_prefetch,
        kernel_type,
        variant,
        enable_pdl,
        out_dtype,
        per_token_alpha,
        l2_policy,
    ) = cache_key
    if kernel_type == "sm100":
        from .kernels.dense_blockscaled_gemm_sm100 import (
            Sm100BlockScaledPersistentDenseGemmKernel,
        )

        return Sm100BlockScaledPersistentDenseGemmKernel(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
            use_prefetch,
            enable_pdl,
            per_token_alpha,
            mma_inst_tile_k=variant or 4,
            a_l2_evict_first=l2_policy == "a_ef",
            b_l2_evict_first=l2_policy == "b_ef",
            a_l2_evict_last=l2_policy == "ab_el",
            b_l2_evict_last=l2_policy == "ab_el",
        )
    if kernel_type == "sm100sk":
        from .kernels.dense_blockscaled_gemm_sm100_splitk import (
            Sm100BlockScaledSplitKGemmKernel,
        )

        return Sm100BlockScaledSplitKGemmKernel(
            sf_vec_size, mma_tiler_mn, variant, enable_pdl, per_token_alpha
        )
    if kernel_type == "sm103":
        from .kernels.dense_blockscaled_gemm_sm103 import (
            Sm103BlockScaledPersistentDenseGemmKernel,
        )

        return Sm103BlockScaledPersistentDenseGemmKernel(
            sf_vec_size, mma_tiler_mn, cluster_shape_mn, variant, enable_pdl
        )
    if kernel_type == "sm107":
        from .kernels.dense_blockscaled_gemm_sm107 import (
            Sm107BlockScaledPersistentDenseGemmKernel,
        )

        return Sm107BlockScaledPersistentDenseGemmKernel(
            sf_vec_size,
            variant[:3],
            (*mma_tiler_mn, variant[3]),
            cluster_shape_mn,
            prefetch_dist=variant[4],
        )
    raise ValueError(f"Unknown FP4 kernel type: {kernel_type}")


def _get_mm_fp4_cute_dsl_compile_workers() -> int:
    """How many subprocesses to use for precompiling mm_fp4 cute-dsl tactics.

    Starts from FLASHINFER_MM_FP4_CUTE_DSL_COMPILE_WORKERS (default 4), then
    lowers it to what host RAM can hold using 1 GiB per spawned worker as a safety measure.
    """
    from ..jit.cute_dsl_core import cap_cute_dsl_compile_workers

    return cap_cute_dsl_compile_workers(
        int(os.environ.get("FLASHINFER_MM_FP4_CUTE_DSL_COMPILE_WORKERS", "4"))
    )


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


def per_token_alpha_mode(per_token_alpha, swap_ab):
    """Kernel extent the alpha vector indexes, or ``None`` for a scalar alpha.

    The SM100 runner swaps A and B at the call site, so under ``swap_ab`` the
    caller's rows are the kernel's N extent.
    """
    if not per_token_alpha:
        return None
    return "n" if swap_ab else "m"


def _mm_fp4_cache_key(
    sf_vec_size,
    tactic,
    enable_pdl,
    out_dtype,
    per_token_alpha=None,
    l2_policy=None,
):
    """In-memory kernel-cache key for one mm_fp4 tactic tuple.

    Shared by the runner's forward path and the precompile path, which
    must agree byte-for-byte: the on-disk kernel name derives from it.
    ``l2_policy`` is :func:`mm_fp4_l2_policy` for the shape.
    """
    return (
        sf_vec_size,
        *tactic,
        enable_pdl,
        out_dtype,
        per_token_alpha,
        l2_policy,
    )


def mm_fp4_l2_policy(m, mma_tiler_mn, swap_ab, kernel_type):
    """L2 eviction policy for the operand streams of an ``"sm100"`` tactic.

    The persistent kernel streams the weight matrix once per token tile while
    every CTA of a column re-reads the same token tile and its scale factors.

    * With at most two token tiles the weights are read at most twice, so
      loading them with ``evict_first`` keeps the shared token tile resident
      instead of letting the 32-117 MB weight stream flush it (1.04-1.24x on
      the M <= 512 rows of the projection shapes, both B200 and GB300).
      Returns ``"a_ef"`` (weights are the kernel's A operand, ``swap_ab``) or
      ``"b_ef"`` (B operand).
    * With more token tiles both operands are re-read through L2 (the
      weights by every M tile, the token tiles by every N tile): loading both
      with ``evict_last`` gains 0.1-10 % on the M = 2048 / 8192 rows of the
      no-swap kernel (32/32 rows > 1 on B200 and GB300; ``evict_last`` on the
      token operand alone left two GB300 rows at 0.999). Returns ``"ab_el"``.

    Returns ``None`` for other kernels. Cache hints only: outputs do not
    change.
    """
    if kernel_type != "sm100":
        return None
    token_tile = mma_tiler_mn[1] if swap_ab else mma_tiler_mn[0]
    if (m + token_tile - 1) // token_tile <= 2:
        return "a_ef" if swap_ab else "b_ef"
    return None if swap_ab else "ab_el"


def precompile_mm_fp4_tactics(
    tactics,
    m,
    n,
    real_k,
    use_nvfp4,
    enable_pdl,
    out_dtype,
    kernel_cache,
    device,
    per_token_alpha=False,
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
        cache_key = _mm_fp4_cache_key(
            sf_vec_size,
            tactic,
            enable_pdl,
            out_dtype,
            per_token_alpha_mode(per_token_alpha, swap_ab),
            mm_fp4_l2_policy(m, mma_tiler_mn, swap_ab, kernel_type),
        )
        if (device_index, cache_key) in kernel_cache:
            continue

        cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
        if cluster_size not in max_clusters_cache:
            max_clusters_cache[cluster_size] = get_max_active_clusters(cluster_size)
        mac = max_clusters_cache[cluster_size]
        kernel_name = _blockscaled_kernel_disk_name(cache_key, 1, None)
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
