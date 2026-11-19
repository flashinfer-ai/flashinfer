# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rubin (SM107) persistent dense BF16 GEMM with direct split-K.

Port of TensorRT-LLM's Rubin BF16 GEMM
(tensorrt_llm/_torch/cute_dsl_kernels/rubin/dense_bf16_gemm_persistent.py and
the split-K path of ``CuteDSLBf16RubinGemmRunner``). It reuses the Blackwell
``PersistentDenseGemmKernel`` and retargets it to sm_107, whose larger shared
memory allows deeper A/B pipelines. It serves ``mm_bf16`` for M > 32 with two
options the autotuner picks per shape:

* swap-AB: kernel A is the weight ``W[N, K]`` and kernel B the activations
  ``X[M, K]``; the kernel writes ``out^T``, an M-major view of ``out``, and
  streams the weight once per MMA tile N of tokens.
* split-K: each output tile is computed by ``split_k`` CTAs over disjoint K
  ranges, and each reduce-adds its BF16 partial into ``out`` with a TMA
  reduction store. ``out`` is zeroed right before the launch on the same
  stream, so CUDA-graph capture and replay stay correct. The summation order
  across splits is not deterministic.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import torch

from .bmm_fp8_blackwell import PersistentDenseGemmKernel
from .bmm_fp8_wrapper import _create_fake_tensors

#: (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn) the autotuner sweeps.
_KERNEL_CONFIGS = (
    (False, (128, 64), (1, 1)),
    (False, (128, 128), (1, 1)),
    (False, (128, 256), (1, 1)),
    (False, (128, 128), (2, 1)),
    (True, (256, 128), (2, 1)),
    (True, (256, 256), (2, 1)),
)

#: K slices per output tile; 1 is the plain (non-reducing) TMA store.
_SUPPORTED_SPLIT_K = (1, 2, 4, 8)


class Sm107PersistentDenseBf16GemmKernel(PersistentDenseGemmKernel):
    """``PersistentDenseGemmKernel`` built for SM107 with BF16/FP16 operands.

    The tcgen05 BF16 MMA is shared by SM100 and SM107, so only the target
    architecture changes: ``_setup_attributes`` sizes the A/B/C stages from
    SM107's shared memory. Always uses the TMA store epilogue (split-K needs it).
    """

    def __init__(
        self,
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        split_k_slices: int = 1,
    ):
        super().__init__(
            cutlass.Float32,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            use_tma_store=True,
            split_k_slices=split_k_slices,
        )
        self.arch = "sm_107"

    def check_supported_dtypes(self, a_dtype, b_dtype, c_dtype):
        """A, B and C share one 16-bit float type."""
        if a_dtype not in {cutlass.BFloat16, cutlass.Float16} or not (
            a_dtype == b_dtype == c_dtype
        ):
            raise testing.CantImplementError(
                f"Expected matching BF16/FP16 A, B and C, got {a_dtype}, "
                f"{b_dtype} and {c_dtype}"
            )
        return super().check_supported_dtypes(a_dtype, b_dtype, c_dtype)

    def can_implement(self, mnkl, a_dtype, b_dtype, c_dtype, a_major, b_major, c_major):
        if not super().can_implement(
            mnkl, a_dtype, b_dtype, c_dtype, a_major, b_major, c_major
        ):
            return False
        # From TensorRT-LLM: an M-cluster wider than the M CTA tiles launches
        # phantom CTAs whose multicast peers never run; on SM107 that is an
        # illegal memory access or a hang.
        cta_tile_m = self.mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1)
        return (mnkl[0] + cta_tile_m - 1) // cta_tile_m >= self.cluster_shape_mn[0]


@dataclasses.dataclass(frozen=True)
class Sm107Bf16Tactic:
    """One kernel specialization for the public ``X[M, K] @ W[N, K]^T``.

    ``swap_ab`` puts W on kernel-M (and tokens on kernel-N); the MMA tile and
    cluster are in kernel coordinates.
    """

    swap_ab: bool
    use_2cta_instrs: bool
    mma_m: int
    mma_n: int
    cluster_m: int
    cluster_n: int
    split_k: int

    def make_kernel(self) -> Sm107PersistentDenseBf16GemmKernel:
        return Sm107PersistentDenseBf16GemmKernel(
            self.use_2cta_instrs,
            (self.mma_m, self.mma_n),
            (self.cluster_m, self.cluster_n),
            self.split_k,
        )


@functools.lru_cache(maxsize=4096)
def _can_implement(tactic: Sm107Bf16Tactic, m: int, n: int, k: int) -> bool:
    # Kernel C is out (N-major) or, with swap-AB, out^T (M-major).
    kernel_m, kernel_n = (n, m) if tactic.swap_ab else (m, n)
    return tactic.make_kernel().can_implement(
        (kernel_m, kernel_n, k, 1),
        cutlass.BFloat16,
        cutlass.BFloat16,
        cutlass.BFloat16,
        "k",
        "k",
        "m" if tactic.swap_ab else "n",
    )


def validate_tactic(tactic: Sm107Bf16Tactic, m: int, n: int, k: int) -> None:
    """Reject a tactic that cannot serve ``X[m, k] @ W[n, k]^T``."""
    config = (
        tactic.use_2cta_instrs,
        (tactic.mma_m, tactic.mma_n),
        (tactic.cluster_m, tactic.cluster_n),
    )
    if config not in _KERNEL_CONFIGS:
        raise ValueError(f"unsupported kernel configuration {config}")
    if tactic.split_k not in _SUPPORTED_SPLIT_K:
        raise ValueError(f"unsupported split_k={tactic.split_k}")
    if not _can_implement(tactic, m, n, k):
        raise ValueError(f"tactic {tactic} cannot implement M={m}, N={n}, K={k}")


def autotune_tactics(m: int, n: int, k: int) -> list[Sm107Bf16Tactic]:
    """Return every valid tactic: orientation x kernel configuration x split-K."""
    tactics = []
    for swap_ab in (True, False):
        for use_2cta_instrs, (mma_m, mma_n), (cluster_m, cluster_n) in _KERNEL_CONFIGS:
            for split_k in _SUPPORTED_SPLIT_K:
                tactic = Sm107Bf16Tactic(
                    swap_ab,
                    use_2cta_instrs,
                    mma_m,
                    mma_n,
                    cluster_m,
                    cluster_n,
                    split_k,
                )
                try:
                    validate_tactic(tactic, m, n, k)
                except ValueError:
                    continue
                tactics.append(tactic)
    return tactics


#: Smallest K range one split-K slice keeps; shorter slices spend more time
#: filling the pipeline and reduce-adding than they save.
_MIN_K_PER_SPLIT = 768


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def default_tactic(m: int, n: int, k: int, device: torch.device) -> Sm107Bf16Tactic:
    """The tactic for calls without autotuning.

    Swap-AB puts the tokens on kernel-N: a 64-token tile up to 64 tokens, a
    128-token tile up to 128, and a 2-CTA 256x128 tile above. Split-K then
    takes the largest factor whose CTAs still fit in one wave, keeping at least
    ``_MIN_K_PER_SPLIT`` of K per slice. Falls back to fewer splits, then to
    the plain 128x128 tile, when a tactic cannot implement the shape.
    """
    sms = _num_sms(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    if m <= 64:
        tiles = [(False, 128, 64, 1), (False, 128, 128, 1)]
    elif m <= 128:
        tiles = [(False, 128, 128, 1)]
    else:
        tiles = [(True, 256, 128, 2), (False, 128, 128, 1)]
    for use_2cta_instrs, mma_m, mma_n, cluster_m in tiles:
        # One CTA covers 128 weight rows (half of a 2-CTA 256-row MMA tile).
        ctas = -(-n // 128) * -(-m // mma_n)
        for split_k in sorted(_SUPPORTED_SPLIT_K, reverse=True):
            if split_k > 1 and (
                ctas * split_k > sms or k // split_k < _MIN_K_PER_SPLIT
            ):
                continue
            tactic = Sm107Bf16Tactic(
                swap_ab=True,
                use_2cta_instrs=use_2cta_instrs,
                mma_m=mma_m,
                mma_n=mma_n,
                cluster_m=cluster_m,
                cluster_n=1,
                split_k=split_k,
            )
            if _can_implement(tactic, m, n, k):
                return tactic
    raise ValueError(f"no SM107 persistent BF16 tactic implements M={m}, N={n}, K={k}")


def validate_inputs(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Check the ``mm_bf16`` layouts: row-major ``a[M, K]`` and ``out[M, N]``,
    column-major ``b[K, N]`` (a row-major weight ``W[N, K]``)."""
    if any(t.dtype != torch.bfloat16 for t in (a, b, out)):
        raise ValueError("a, b and out must be bfloat16")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("a, b and out must be 2D")
    if b.device != a.device or out.device != a.device:
        raise ValueError("a, b and out must be on the same device")
    m, k = a.shape
    n = b.shape[1]
    if b.shape[0] != k or out.shape != (m, n):
        raise ValueError(
            f"incompatible shapes: a {tuple(a.shape)}, b {tuple(b.shape)}, "
            f"out {tuple(out.shape)}"
        )
    if not (a.is_contiguous() and b.T.is_contiguous() and out.is_contiguous()):
        raise ValueError("a and out must be row-major and b column-major")
    # TMA: 16-byte aligned bases and row strides (out^T needs N % 8 too).
    if k % 8 or n % 8:
        raise ValueError(f"K={k} and N={n} must be multiples of 8")
    if any(t.data_ptr() % 16 for t in (a, b, out)):
        raise ValueError("a, b and out must be 16-byte aligned")
    # All three are dense, so each covers one contiguous byte range.
    out_begin = out.data_ptr()
    out_end = out_begin + out.numel() * out.element_size()
    for t in (a, b):
        begin = t.data_ptr()
        if begin < out_end and out_begin < begin + t.numel() * t.element_size():
            raise ValueError("out must not overlap a or b")


@cute.jit
def _bmm_bf16(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # (l, m, k)
    b: cute.Tensor,  # (l, k, n)
    c: cute.Tensor,  # (l, m, n)
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """``bmm`` from bmm_fp8_blackwell.py without the FP8 output scale."""
    # (l,m,k) -> (m,k,l)
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    # (l,k,n) -> (n,k,l)
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    # (l,m,n) -> (m,n,l)
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(a, b, c, max_active_clusters, stream, None, None)


@functools.cache
def _get_compiled_kernel(tactic: Sm107Bf16Tactic):
    """Compile once per tactic; M, N and K are symbolic."""
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    a_fake, b_fake, c_fake, _, _, stream_fake = _create_fake_tensors(
        cutlass.BFloat16,
        cutlass.BFloat16,
        "k",
        "k",
        "m" if tactic.swap_ab else "n",
    )
    # Options as in TensorRT-LLM's runner, where the VR200 timings come from.
    return cute.compile(
        _bmm_bf16,
        tactic.make_kernel(),
        a_fake,
        b_fake,
        c_fake,
        get_max_active_clusters(tactic.cluster_m * tactic.cluster_n),
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def _as_batch1(t: torch.Tensor) -> torch.Tensor:
    """View a 2D tensor as ``(1, *shape)`` with a compact batch stride."""
    return t.as_strided((1, *t.shape), (t.numel(), *t.stride()))


def run_persistent_dense(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    tactic: Sm107Bf16Tactic,
) -> torch.Tensor:
    """Run ``out = a[M, K] @ b[K, N]`` with the ``mm_bf16`` layouts."""
    validate_inputs(a, b, out)
    m, k = a.shape
    n = b.shape[1]
    validate_tactic(tactic, m, n, k)
    if tactic.swap_ab:
        # A = W[N, K], B = X[M, K] (as (k, n) = X^T), C = out^T[N, M]
        tensors = (_as_batch1(b.T), _as_batch1(a.T), _as_batch1(out.T))
    else:
        # A = X[M, K], B = W^T (k, n), C = out[M, N]
        tensors = (_as_batch1(a), _as_batch1(b), _as_batch1(out))
    with torch.cuda.device(a.device):
        compiled = _get_compiled_kernel(tactic)
        if tactic.split_k > 1:
            # The K slices reduce-add into out. Zero it on the launch stream
            # right before the GEMM so graph replays start from zero too.
            out.zero_()
        compiled(*tensors)
    return out


__all__ = [
    "Sm107Bf16Tactic",
    "Sm107PersistentDenseBf16GemmKernel",
    "autotune_tactics",
    "default_tactic",
    "run_persistent_dense",
    "validate_inputs",
    "validate_tactic",
]
