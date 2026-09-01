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
from dataclasses import replace
from typing import List, Literal, Optional, Tuple

from packaging.version import Version
import torch

from ..api_logging import flashinfer_api
from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..fused_moe.utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from ..jit.cpp_ext import get_cuda_version
from ..jit.gemm import gen_gemm_sm100_module_cutlass_nvfp4_svdquant
from ..trace.templates.gemm import (
    mm_nvfp4_svdquant_trace,
    nvfp4_quantize_smooth_trace,
    svdquant_linear_trace,
)
from ..utils import (
    _get_cache_buf,
    backend_requirement,
    device_support_pdl,
    get_device_sm_count,
    supported_compute_capability,
)

DEFAULT_WORKSPACE_SIZE = 32 * 1024 * 1024

# The fused kernel accumulates the rank-r BF16 LoRA-up into the NVFP4 residual accumulator.
# The rank is inferred from the d/l1 shapes and must be a positive multiple of the collective's
# rank granularity (CollectiveMmaLoRA::LoRaK); ranks 32-128 are validated.
SVDQUANT_LORA_RANK_GRANULARITY = 32

_SM120_SVDQUANT_KERNEL_CACHE: dict[tuple, object] = {}
_MIN_SM120_SVDQUANT_CUDA_VERSION = Version("12.9")


def _pad_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _swizzled_sf_size(rows: int, sf_cols: int) -> int:
    """Size of the 128x4-swizzled block-scale layout for a [rows, sf_cols] scale matrix."""
    return _pad_up(rows, 128) * _pad_up(sf_cols, 4)


def _view_128x4_sf(sf: torch.Tensor, rows: int, sf_cols: int) -> torch.Tensor:
    """Restore a flat public scale buffer to its padded 128x4 storage view."""
    size = _swizzled_sf_size(rows, sf_cols)
    return sf.reshape(-1)[:size].view(_pad_up(rows, 128), _pad_up(sf_cols, 4))


def _svdquant_kernel_source_files() -> Tuple[str, ...]:
    """Sources whose device-code changes invalidate the SM120 disk cache."""
    from ..cute_dsl import utils as cute_dsl_utils
    from .kernels import dense_blockscaled_gemm_sm120_b12x

    return (
        __file__,
        dense_blockscaled_gemm_sm120_b12x.__file__,
        cute_dsl_utils.__file__,
    )


def _sm120_svdquant_kernel_name(
    *,
    rank: int,
    with_bias: bool,
    mma_tiler_mn: Tuple[int, int],
    tile_k: int,
    swap_ab: bool,
    max_active_clusters: int,
    enable_pdl: bool,
    enable_iket: bool,
) -> str:
    """Return a symbol-safe name encoding every SM120 codegen parameter."""
    return (
        f"r{rank}_bias{int(with_bias)}_t{mma_tiler_mn[0]}x{mma_tiler_mn[1]}"
        f"x{tile_k}_swap{int(swap_ab)}_mac{max_active_clusters}"
        f"_pdl{int(enable_pdl)}_iket{int(enable_iket)}"
    )


def _compile_sm120_nvfp4_svdquant(
    *,
    device: torch.device,
    rank: int,
    with_bias: bool,
    mma_tiler_mn: Tuple[int, int],
    tile_k: int,
    swap_ab: bool,
    sf_m: int,
    sf_n: int,
    sf_k: int,
    enable_pdl: bool,
    enable_iket: bool = False,
):
    """Compile one fused SM120 NVFP4 + rank-r BF16 epilogue specialization."""
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    from ..cute_dsl.utils import get_max_active_clusters

    max_active_clusters = get_max_active_clusters(1)
    cache_key = (
        device_index,
        rank,
        with_bias,
        mma_tiler_mn,
        tile_k,
        swap_ab,
        max_active_clusters,
        enable_pdl,
        enable_iket,
    )
    if cache_key in _SM120_SVDQUANT_KERNEL_CACHE:
        return _SM120_SVDQUANT_KERNEL_CACHE[cache_key]

    import cutlass
    import cutlass.cute as cute

    from cutlass.cute.runtime import make_ptr

    from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from .kernels.dense_blockscaled_gemm_sm120_b12x import (
        Sm120B12xBlockScaledDenseGemmKernel,
    )

    gemm = Sm120B12xBlockScaledDenseGemmKernel(
        16,
        mma_tiler_mn,
        (1, 1),
        tile_k=tile_k,
        # The shared b12x constructor retains this generic-GEMM knob, but the
        # SM120 SVDQuant kernel has no prefetch dataflow specialization.
        use_prefetch=False,
        enable_pdl=enable_pdl,
        swap_ab=swap_ab,
        enable_iket=enable_iket,
    )

    def compile_kernel():
        sym_m = cute.sym_int()
        sym_k = cute.sym_int()
        sym_n = cute.sym_int()
        a_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (sym_m, sym_k),
            stride_order=(1, 0),
            assumed_align=32,
        )
        b_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (sym_n, sym_k),
            stride_order=(1, 0),
            assumed_align=32,
        )
        c_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_m, sym_n),
            stride_order=(1, 0),
            assumed_align=16,
        )
        d_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_m, rank),
            stride_order=(1, 0),
            assumed_align=16,
        )
        l1_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16,
            (sym_n, rank),
            stride_order=(1, 0),
            assumed_align=16,
        )
        bias_fake = (
            cute.runtime.make_fake_compact_tensor(
                cutlass.BFloat16, (sym_n,), assumed_align=16
            )
            if with_bias
            else None
        )
        a_sf_ptr = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16)
        b_sf_ptr = make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16)
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
            1,
            a_sf_ptr,
            b_sf_ptr,
            alpha_fake,
            max_active_clusters,
            stream_fake,
            False,
            d_fake,
            l1_fake,
            bias_fake,
            options="--opt-level 2 --enable-tvm-ffi",
        )

    kernel_name = _sm120_svdquant_kernel_name(
        rank=rank,
        with_bias=with_bias,
        mma_tiler_mn=mma_tiler_mn,
        tile_k=tile_k,
        swap_ab=swap_ab,
        max_active_clusters=max_active_clusters,
        enable_pdl=enable_pdl,
        enable_iket=enable_iket,
    )
    compiled = build_and_load_cute_dsl_kernel(
        "mm_nvfp4_svdquant_sm120",
        kernel_name,
        compile_kernel,
        extra_key_files=_svdquant_kernel_source_files(),
    )
    _SM120_SVDQUANT_KERNEL_CACHE[cache_key] = compiled
    return compiled


def _mm_nvfp4_svdquant_sm120_fused(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    enable_pdl: bool,
    tactic=None,
    enable_iket: bool = False,
) -> torch.Tensor:
    from .kernels.dense_blockscaled_gemm_sm120_b12x import (
        _select_default_dense_gemm_plan,
    )

    m, k_packed = a.shape
    n = b.shape[0]
    real_k = k_packed * 2
    sf_m = (m + 127) // 128
    sf_n = (n + 127) // 128
    sf_k = (real_k // 16 + 3) // 4
    if tactic is None or tactic == -1:
        plan = _select_default_dense_gemm_plan(
            m, n, real_k, get_device_sm_count(a.device), expected_m=m
        )
        tactic = (plan.mma_tiler_mn, 128, plan.swap_ab)
    mma_tiler_mn, tile_k, swap_ab = tactic
    compiled = _compile_sm120_nvfp4_svdquant(
        device=a.device,
        rank=d.shape[1],
        with_bias=bias is not None,
        mma_tiler_mn=mma_tiler_mn,
        tile_k=tile_k,
        swap_ab=swap_ab,
        sf_m=sf_m,
        sf_n=sf_n,
        sf_k=sf_k,
        enable_pdl=enable_pdl,
        enable_iket=enable_iket,
    )
    args = [
        a,
        b,
        out,
        sf_m,
        sf_n,
        sf_k,
        a_sf.data_ptr(),
        b_sf.data_ptr(),
        alpha,
        d,
        l1,
        bias,
    ]
    compiled(*args)
    return out


def _mm_nvfp4_svdquant_sm120_unfused(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    enable_pdl: bool,
) -> torch.Tensor:
    from .gemm_base import mm_fp4

    m, k_packed = a.shape
    n = b.shape[0]
    sf_cols = k_packed * 2 // 16
    a_sf_2d = _view_128x4_sf(a_sf, m, sf_cols)
    b_sf_2d = _view_128x4_sf(b_sf, n, sf_cols)
    mm_fp4(
        a,
        b.T,
        a_sf_2d,
        b_sf_2d.T,
        alpha,
        torch.bfloat16,
        out,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="b12x",
        use_nvfp4=True,
        enable_pdl=enable_pdl,
    )
    correction = torch.mm(d, l1.T)
    correction.mul_(alpha)
    out.add_(correction)
    if bias is not None:
        out.add_(bias)
    return out


def _sm120_nvfp4_svdquant_runner(enable_pdl: bool):
    class Sm120Nvfp4SvdquantRunner(TunableRunner):
        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            return (enable_pdl,)

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> list:
            import cutlass

            from ..cute_dsl.utils import torch_to_cutlass_dtype
            from .kernels.dense_blockscaled_gemm_sm120_b12x import (
                Sm120B12xBlockScaledDenseGemmKernel,
                _select_default_dense_gemm_plan,
            )

            a, b, _, _, _, d, _, _, out = inputs
            m, k_packed = a.shape
            n = b.shape[0]
            real_k = k_packed * 2
            c_dtype = torch_to_cutlass_dtype(out.dtype)
            tactics = []

            def _add(
                mma_tiler_mn,
                tile_k,
                swap_ab,
            ):
                if not Sm120B12xBlockScaledDenseGemmKernel.can_implement(
                    cutlass.Float4E2M1FN,
                    cutlass.Float8E4M3FN,
                    16,
                    c_dtype,
                    mma_tiler_mn,
                    (1, 1),
                    n,
                    real_k,
                    1,
                    "k",
                    "k",
                    "n",
                    swap_ab=swap_ab,
                    svdquant_rank=d.shape[1],
                    tile_k=tile_k,
                ):
                    return
                tactic = (mma_tiler_mn, tile_k, swap_ab)
                if tactic not in tactics:
                    tactics.append(tactic)

            for mma_tiler_mn in ((64, 64), (64, 128), (128, 64), (128, 128)):
                _add(mma_tiler_mn, 128, swap_ab=False)

            plan = _select_default_dense_gemm_plan(
                m, n, real_k, get_device_sm_count(a.device), expected_m=m
            )
            _add(plan.mma_tiler_mn, 128, plan.swap_ab)
            for tile_k in (64, 256):
                _add(plan.mma_tiler_mn, tile_k, plan.swap_ab)
            if m >= 256 and n >= 64:
                _add((256, 64), 128, swap_ab=False)
            return tactics

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic=None,
            do_preparation: bool = False,
            **kwargs,
        ):
            a, b, a_sf, b_sf, alpha, d, l1, bias, out = inputs
            return _mm_nvfp4_svdquant_sm120_fused(
                a,
                b,
                a_sf,
                b_sf,
                alpha,
                d,
                l1,
                bias,
                out,
                enable_pdl,
                tactic,
            )

    return Sm120Nvfp4SvdquantRunner()


def _sm120_nvfp4_svdquant_unfused_runner(enable_pdl: bool):
    # Flatten the b12x runner into the outer fused-vs-unfused tuner. Calling
    # mm_fp4 here would start a nested AutoTuner while the outer runner is under
    # CUDA Graph capture; on a cold cache that leaves tensor initialization in
    # the capture and fails before the unfused candidate can be profiled.
    from .gemm_base import _b12x_gemm_fp4_runner

    fp4_runner = _b12x_gemm_fp4_runner(
        12,
        0,
        enable_pdl,
        torch.bfloat16,
        True,
    )
    workspace_buffers: dict[torch.device, torch.Tensor] = {}

    class Sm120Nvfp4SvdquantUnfusedRunner(TunableRunner):
        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            return (enable_pdl,)

        def _fp4_inputs(self, inputs: List[torch.Tensor]) -> list:
            a, b, a_sf, b_sf, alpha, _, _, _, out = inputs
            m, k_packed = a.shape
            n = b.shape[0]
            sf_cols = k_packed * 2 // 16
            workspace_buffer = workspace_buffers.get(a.device)
            if workspace_buffer is None:
                workspace_buffer = _get_cache_buf(
                    "mm_fp4_workspace",
                    DEFAULT_WORKSPACE_SIZE,
                    a.device,
                )
                workspace_buffers[a.device] = workspace_buffer
            return [
                a,
                b.T,
                _view_128x4_sf(a_sf, m, sf_cols),
                _view_128x4_sf(b_sf, n, sf_cols).T,
                alpha,
                torch.bfloat16,
                out,
                16,
                True,
                workspace_buffer,
            ]

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> list:
            return fp4_runner.get_valid_tactics(self._fp4_inputs(inputs), profile)

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic=-1,
            do_preparation: bool = False,
            **kwargs,
        ):
            _, _, _, _, alpha, d, l1, bias, out = inputs
            fp4_runner(inputs=self._fp4_inputs(inputs), tactic=tactic)
            correction = torch.mm(d, l1.T)
            correction.mul_(alpha)
            out.add_(correction)
            if bias is not None:
                out.add_(bias)
            return out

    return Sm120Nvfp4SvdquantUnfusedRunner()


_SM120_NVFP4_SVDQUANT_TUNING_CONFIG = TuningConfig(
    use_cuda_graph=True,
    use_cold_l2_cache=True,
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),
            (0,),
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        ),
    ),
    constraint_specs=(
        ConstraintSpec(
            2,
            0,
            lambda shapes: _swizzled_sf_size(shapes[0][0], shapes[0][1] * 2 // 16),
        ),
        ConstraintSpec(5, 0, lambda shapes: shapes[0][0]),
        ConstraintSpec(8, 0, lambda shapes: shapes[0][0]),
    ),
)

# The unfused candidate is a composition of FP4 GEMM, BF16 GEMM, and pointwise
# kernels. Profiling it inside the same CUDA Graph tuner as the single fused
# kernel can leave capture active after a failed tactic. Keep graph profiling
# for fused tactic selection, but compare fused versus unfused in eager mode.
_SM120_NVFP4_SVDQUANT_IMPLEMENTATION_TUNING_CONFIG = replace(
    _SM120_NVFP4_SVDQUANT_TUNING_CONFIG,
    use_cuda_graph=False,
)


@functools.cache
def get_nvfp4_svdquant_module():
    """JIT-build and load the SM100 CUTLASS NVFP4 SVDQuant module."""
    return gen_gemm_sm100_module_cutlass_nvfp4_svdquant().build_and_load()


def _nvfp4_svdquant_gemm_runner(enable_pdl: bool):
    module = get_nvfp4_svdquant_module()

    class Nvfp4SvdquantGemmRunner(TunableRunner):
        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            return (enable_pdl,)

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            return list(range(module.nvfp4_svdquant_gemm_tactic_num()))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ):
            (a, b, a_sf, b_sf, alpha, d, l1, bias, out, workspace_buffer) = inputs
            module.nvfp4_svdquant_gemm(
                a,
                b,
                a_sf,
                b_sf,
                alpha,
                d,
                l1,
                bias,
                out,
                workspace_buffer,
                tactic,
                enable_pdl,
            )
            return out

    return Nvfp4SvdquantGemmRunner()


_NVFP4_SVDQUANT_GEMM_TUNING_CONFIG = TuningConfig(
    use_cuda_graph=True,
    use_cold_l2_cache=True,
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (0,),
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        ),
    ),
    constraint_specs=(
        ConstraintSpec(
            2,  # a_sf tensor index: 1-D 128x4-swizzled scale buffer sized by (m, k/16)
            0,
            lambda shapes: _swizzled_sf_size(shapes[0][0], shapes[0][1] * 2 // 16),
        ),
        ConstraintSpec(
            5,  # d tensor index: [m, r] LoRA-down output (r kept from the real input)
            0,
            lambda shapes: shapes[0][0],
        ),
        ConstraintSpec(
            8,  # out tensor index
            0,
            lambda shapes: shapes[0][0],
        ),
        ConstraintSpec(
            9,  # workspace_buffer index: scratch; exclude its (resizable) size from the
            0,  # cache key so a mid-tune resize never causes a silent cache miss.
            lambda shapes: shapes[9][0],
        ),
    ),
)


@supported_compute_capability([100, 103, 107])
def _cutlass_nvfp4_svdquant_requirement(*args, **kwargs):
    return True


@supported_compute_capability([120, 121])
def _cute_dsl_nvfp4_svdquant_requirement(*args, **kwargs):
    cuda_version = get_cuda_version()
    if cuda_version < _MIN_SM120_SVDQUANT_CUDA_VERSION:
        raise ValueError(
            "SM120 SVDQuant CuTe DSL support requires CUDA 12.9 or later. "
            f"Current CUDA version: {cuda_version}."
        )
    from ..cute_dsl import is_cute_dsl_available

    if not is_cute_dsl_available():
        raise ValueError(
            "SM120 SVDQuant CuTe DSL support requires CuTe DSL, but it is not "
            "available in the current environment."
        )
    return True


def _heuristic_func_nvfp4_svdquant(
    suitable_backends: List[str], *args, **kwargs
) -> List[str]:
    # Preserve backend_checks order: on SM120/SM121, cute-dsl precedes
    # cute-dsl-unfused when both are supported, which selects the fused-first
    # implementation tuning configuration.
    return suitable_backends


def _check_mm_nvfp4_svdquant_problem(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cutlass", "cute-dsl", "cute-dsl-unfused", "auto"] = "auto",
    enable_pdl: Optional[bool] = None,
):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2-D packed-e2m1 (uint8) tensors")
    if a.dtype != torch.uint8 or b.dtype != torch.uint8:
        raise ValueError("a and b must be uint8 (two e2m1 values per byte)")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous")
    m, k_packed = a.shape
    n = b.shape[0]
    if b.shape[1] != k_packed:
        raise ValueError(
            f"a and b inner dimensions mismatch: a {tuple(a.shape)} vs b {tuple(b.shape)}"
        )
    k = k_packed * 2
    if n % 32 != 0 or k % 32 != 0:
        raise ValueError(f"n and k must be divisible by 32, got n={n}, k={k}")
    if d.ndim != 2 or d.shape[0] != m:
        raise ValueError(
            f"d must have shape [m, r] (rank-r LoRA-down output), got {tuple(d.shape)}"
        )
    rank = d.shape[1]
    if rank < SVDQUANT_LORA_RANK_GRANULARITY or rank % SVDQUANT_LORA_RANK_GRANULARITY:
        raise ValueError(
            f"the LoRA rank (d.shape[1]) must be a positive multiple of "
            f"{SVDQUANT_LORA_RANK_GRANULARITY}, got {rank}"
        )
    if l1.ndim != 2 or l1.shape[0] != n or l1.shape[1] != rank:
        raise ValueError(
            f"l1 must have shape [n, {rank}] (rank-{rank} LoRA-up weight pre-divided "
            f"by alpha, same rank as d), got {tuple(l1.shape)}"
        )
    if d.dtype != torch.bfloat16 or l1.dtype != torch.bfloat16:
        raise ValueError("d and l1 must be bf16")
    if not d.is_contiguous() or not l1.is_contiguous():
        raise ValueError("d and l1 must be contiguous")
    if a_sf.dtype != torch.uint8 or b_sf.dtype != torch.uint8:
        raise ValueError("a_sf and b_sf must be uint8 (ue4m3 block scales)")
    expected_a_sf = _swizzled_sf_size(m, k // 16)
    expected_b_sf = _swizzled_sf_size(n, k // 16)
    if a_sf.numel() < expected_a_sf or b_sf.numel() < expected_b_sf:
        raise ValueError(
            "128x4 scale buffers are too small: "
            f"a_sf has {a_sf.numel()} elements (need {expected_a_sf}), "
            f"b_sf has {b_sf.numel()} elements (need {expected_b_sf})"
        )
    if not a_sf.is_contiguous() or not b_sf.is_contiguous():
        raise ValueError("a_sf and b_sf must be contiguous")
    if alpha.dtype != torch.float32 or alpha.numel() < 1:
        raise ValueError("alpha must be a non-empty float32 device tensor")
    if bias is not None and (bias.shape != (n,) or bias.dtype != torch.bfloat16):
        raise ValueError(f"bias must have shape ({n},) and dtype bf16")
    if out is not None:
        if out.shape != (m, n) or out.dtype != torch.bfloat16:
            raise ValueError(f"out must have shape ({m}, {n}) and dtype bf16")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
    tensors = (b, a_sf, b_sf, alpha, d, l1, bias, out)
    if any(t is not None and t.device != a.device for t in tensors):
        raise ValueError("all SVDQuant tensors must be on the same device")
    return True


@backend_requirement(
    {
        "cutlass": _cutlass_nvfp4_svdquant_requirement,
        "cute-dsl": _cute_dsl_nvfp4_svdquant_requirement,
        "cute-dsl-unfused": _cute_dsl_nvfp4_svdquant_requirement,
    },
    common_check=_check_mm_nvfp4_svdquant_problem,
    heuristic_func=_heuristic_func_nvfp4_svdquant,
)
@flashinfer_api(trace=mm_nvfp4_svdquant_trace)
def mm_nvfp4_svdquant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cutlass", "cute-dsl", "cute-dsl-unfused", "auto"] = "auto",
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""SVDQuant NVFP4 GEMM: ``out = alpha * (a @ bᵀ + d @ l1ᵀ) [+ bias]``.

    On SM100/SM103, CUTLASS fuses the block-scaled NVFP4 residual GEMM with the rank-r
    BF16 LoRA-up correction and optional bias. On SM120/SM121, ``"cute-dsl"`` fuses the
    correction and bias into the b12x CuTe DSL kernel's FP32 accumulator epilogue, while
    ``"cute-dsl-unfused"`` retains the compositional implementation as a
    differential oracle and optional autotuning candidate. The LoRA rank ``r`` is inferred
    from the ``d``/``l1`` shapes and must
    be a positive multiple of 32 (ranks 32-128 are validated). ``1/alpha`` must be folded
    into ``l1`` by the caller (``l1 = svdquant_lora_b / alpha``), so both backends yield
    the correction at its original scale.

    Parameters
    ----------
    a: torch.Tensor
        Quantized activation, shape ``(m, k // 2)`` uint8 (packed e2m1), row-major. Produce it
        with :func:`nvfp4_quantize_smooth` (which folds the SVDQuant ``pre_quant_scale`` into
        the quantization).
    b: torch.Tensor
        Quantized residual weight, shape ``(n, k // 2)`` uint8 (packed e2m1), row-major
        (i.e. the GEMM computes ``a @ bᵀ``).
    a_sf: torch.Tensor
        Activation block scales, uint8 (ue4m3) in the 128x4 swizzled layout,
        ``numel >= ceil(m / 128) * 128 * ceil(k / 16 / 4) * 4``.
    b_sf: torch.Tensor
        Weight block scales, same layout as ``a_sf`` with ``n`` rows.
    alpha: torch.Tensor
        Per-tensor residual dequantization scale in a non-empty float32 device
        tensor. For compatibility with pooled scalar buffers, only the first
        element is consumed; backend runners receive a one-element view.
    d: torch.Tensor
        LoRA-down output ``x_hat @ L2ᵀ``, shape ``(m, r)`` bf16, contiguous and 16-byte
        aligned (TMA). Compute it as ``x @ (pre_quant_scale[:, None] * L2ᵀ)`` in bf16.
    l1: torch.Tensor
        LoRA-up weight pre-divided by alpha, shape ``(n, r)`` bf16 (same rank as ``d``).
    bias: Optional[torch.Tensor]
        Optional per-column bias, shape ``(n,)`` bf16. Fused by CUTLASS and the
        SM120/SM121 CuTe DSL kernel.
    out: Optional[torch.Tensor]
        Output tensor, shape ``(m, n)`` bf16; allocated when ``None``.
    backend: Literal["cutlass", "cute-dsl", "cute-dsl-unfused", "auto"]
        ``"cutlass"`` selects the fused SM100/SM103 implementation;
        ``"cute-dsl"`` selects the fused SM120/SM121 implementation;
        ``"cute-dsl-unfused"`` selects its compositional reference path;
        ``"auto"`` (default) selects by compute capability. On SM120/SM121,
        fused and unfused are compared only while autotuning is enabled;
        otherwise the fused-first runner is selected.
    enable_pdl: Optional[bool]
        Whether to launch with Programmatic Dependent Launch. Defaults to the device default.

    Returns
    -------
    out: torch.Tensor
        Output tensor, shape ``(m, n)`` bf16.
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(a.device)
    if out is None:
        out = torch.empty(a.shape[0], b.shape[0], dtype=torch.bfloat16, device=a.device)
    # Preserve the historical public numel>=1 contract while specializing all
    # backend kernels and autotune keys on one scalar device element.
    alpha_scalar = alpha.reshape(-1)[:1]

    tune_sm120_implementations = False
    if backend == "auto":
        backend = mm_nvfp4_svdquant.suitable_auto_backends[0]
        tune_sm120_implementations = backend == "cute-dsl"

    if backend == "cute-dsl":
        inputs = [a, b, a_sf, b_sf, alpha_scalar, d, l1, bias, out]
        runners = [_sm120_nvfp4_svdquant_runner(enable_pdl)]
        custom_op = "nvfp4_svdquant_gemm_sm120"
        if tune_sm120_implementations:
            runners.append(_sm120_nvfp4_svdquant_unfused_runner(enable_pdl))
            custom_op = "nvfp4_svdquant_gemm_sm120_auto"
        tuning_config = (
            _SM120_NVFP4_SVDQUANT_IMPLEMENTATION_TUNING_CONFIG
            if tune_sm120_implementations
            else _SM120_NVFP4_SVDQUANT_TUNING_CONFIG
        )
        runner, tactic = AutoTuner.get().choose_one(
            custom_op,
            runners,
            tuning_config,
            inputs,
        )
        runner(inputs=inputs, tactic=tactic)
        return out

    if backend == "cute-dsl-unfused":
        return _mm_nvfp4_svdquant_sm120_unfused(
            a,
            b,
            a_sf,
            b_sf,
            alpha_scalar,
            d,
            l1,
            bias,
            out,
            enable_pdl,
        )

    workspace_buffer = _get_cache_buf(
        "nvfp4_svdquant_gemm_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )

    tuner = AutoTuner.get()
    runners = [_nvfp4_svdquant_gemm_runner(enable_pdl)]
    inputs = [a, b, a_sf, b_sf, alpha_scalar, d, l1, bias, out, workspace_buffer]
    runner, tactic = tuner.choose_one(
        "nvfp4_svdquant_gemm",
        runners,
        _NVFP4_SVDQUANT_GEMM_TUNING_CONFIG,
        inputs,
    )
    runner(inputs=inputs, tactic=tactic)
    return out


def _check_nvfp4_quantize_smooth_problem(
    x: torch.Tensor,
    pre_quant_scale: torch.Tensor,
    global_scale: torch.Tensor,
    enable_pdl: Optional[bool] = None,
    backend: Literal["cutlass", "cute-dsl", "auto"] = "auto",
):
    if x.ndim != 2:
        raise ValueError(f"x must be [m, n], got {tuple(x.shape)}")
    if x.dtype != torch.bfloat16 or pre_quant_scale.dtype != torch.bfloat16:
        raise ValueError("x and pre_quant_scale must be bf16")
    if x.shape[1] % 16 != 0:
        raise ValueError(
            f"n must be divisible by 16 (NVFP4 SF vector size), got {x.shape[1]}"
        )
    if pre_quant_scale.numel() != x.shape[1]:
        raise ValueError(
            f"pre_quant_scale must have n={x.shape[1]} elements, got {pre_quant_scale.numel()}"
        )
    return True


@backend_requirement(
    {
        "cutlass": _cutlass_nvfp4_svdquant_requirement,
        "cute-dsl": _cute_dsl_nvfp4_svdquant_requirement,
    },
    common_check=_check_nvfp4_quantize_smooth_problem,
    heuristic_func=_heuristic_func_nvfp4_svdquant,
)
@flashinfer_api(trace=nvfp4_quantize_smooth_trace)
def nvfp4_quantize_smooth(
    x: torch.Tensor,
    pre_quant_scale: torch.Tensor,
    global_scale: torch.Tensor,
    enable_pdl: Optional[bool] = None,
    backend: Literal["cutlass", "cute-dsl", "auto"] = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Smooth + NVFP4 quantize: ``(xq, sf) = nvfp4-quantize(x * pre_quant_scale)``.

    The SM100/SM103 CUTLASS backend applies the SVDQuant per-input-channel smoothing scale
    and NVFP4-quantizes in one pass. The SM120/SM121 CuTe DSL backend also
    applies smoothing inside the NVFP4 quantizer, avoiding a BF16 intermediate.
    Both use ue4m3 block scales, the 128x4 swizzled layout, and SF vector size 16.

    Parameters
    ----------
    x: torch.Tensor
        Input activation, shape ``(m, n)`` bf16.
    pre_quant_scale: torch.Tensor
        Per-input-channel smoothing scale, shape ``(n,)`` bf16.
    global_scale: torch.Tensor
        Global scale, float32 device scalar: ``(448 * 6) / (x * pre_quant_scale).abs().max()``.
    enable_pdl: Optional[bool]
        Whether to launch with Programmatic Dependent Launch. Defaults to the device default.
    backend: Literal["cutlass", "cute-dsl", "auto"]
        ``"cutlass"`` selects fused smoothing and quantization on SM100/SM103;
        ``"cute-dsl"`` selects fused smoothing plus CuTe DSL quantization on
        SM120/SM121; ``"auto"`` (default) selects by compute capability.

    Returns
    -------
    xq: torch.Tensor
        Quantized tensor, shape ``(m, n // 2)`` uint8 (packed e2m1).
    sf: torch.Tensor
        Block scales, uint8 (ue4m3), 128x4 swizzled layout, 1-D of size
        ``ceil(m / 128) * 128 * ceil(n / 16 / 4) * 4``.
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(x.device)
    if backend == "auto":
        backend = nvfp4_quantize_smooth.suitable_auto_backends[0]
    # Both backends consume vectorized BF16 operands. A contiguous storage-offset
    # view can still be misaligned, so materialize only the exceptional case.
    x = x.contiguous()
    pre_quant_scale = pre_quant_scale.reshape(x.shape[1]).contiguous()
    if x.data_ptr() % 16 != 0:
        x = x.clone()
    if pre_quant_scale.data_ptr() % 16 != 0:
        pre_quant_scale = pre_quant_scale.clone()
    if backend == "cute-dsl":
        from ..quantization.kernels.nvfp4_quantize import (
            nvfp4_quantize_smooth_cute_dsl,
        )

        xq, sf = nvfp4_quantize_smooth_cute_dsl(
            x,
            pre_quant_scale,
            global_scale,
            enable_pdl=enable_pdl,
        )
        return xq.view(torch.uint8), sf.view(torch.uint8).reshape(-1)

    m, n = x.shape
    module = get_nvfp4_svdquant_module()
    xq = torch.empty(m, n // 2, dtype=torch.uint8, device=x.device)
    sf = torch.empty(_swizzled_sf_size(m, n // 16), dtype=torch.uint8, device=x.device)
    module.nvfp4_quantize_smooth(x, pre_quant_scale, global_scale, xq, sf, enable_pdl)
    return xq, sf


@flashinfer_api(trace=svdquant_linear_trace)
def svdquant_linear(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    pre_quant_scale: torch.Tensor,
    l2t_smoothed: torch.Tensor,
    l1_scaled: torch.Tensor,
    global_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
    backend: Literal["cutlass", "cute-dsl", "cute-dsl-unfused", "auto"] = "auto",
) -> torch.Tensor:
    r"""The full SVDQuant linear operator: ``y = x_hat @ (R + L1 @ L2)ᵀ [+ bias]`` where
    ``x_hat = x * pre_quant_scale`` and ``R`` is the NVFP4-quantized residual weight.

    Runs the three-step chain this library's kernels are designed for:

    1. ``xq, x_sf = nvfp4_quantize_smooth(x, pre_quant_scale, global_scale)``
    2. ``down = x @ l2t_smoothed``  (BF16 tensor-core GEMM;
       ``l2t_smoothed = pre_quant_scale[:, None] * L2ᵀ``)
    3. ``mm_nvfp4_svdquant(xq, weight_fp4, x_sf, weight_sf, alpha, down, l1_scaled, bias)``

    The invariant per-layer transforms must be prepared offline by the caller:
    ``l2t_smoothed = (pre_quant_scale[:, None] * svdquant_lora_a.T).to(bf16)`` with shape
    ``(k, r)`` and ``l1_scaled = (svdquant_lora_b / alpha).to(bf16)`` with shape ``(n, r)``,
    where the LoRA rank ``r`` is a positive multiple of 32.

    Parameters
    ----------
    x: torch.Tensor
        Input activation, shape ``(m, k)`` bf16.
    weight_fp4: torch.Tensor
        NVFP4 residual weight, shape ``(n, k // 2)`` uint8 (packed e2m1).
    weight_sf: torch.Tensor
        Weight block scales, uint8 (ue4m3), 128x4 swizzled layout.
    alpha: torch.Tensor
        Per-tensor residual dequantization scale, float32 device scalar.
    pre_quant_scale: torch.Tensor
        Per-input-channel smoothing scale, shape ``(k,)`` bf16.
    l2t_smoothed: torch.Tensor
        ``pre_quant_scale[:, None] * L2ᵀ``, shape ``(k, r)`` bf16.
    l1_scaled: torch.Tensor
        ``L1 / alpha``, shape ``(n, r)`` bf16.
    global_scale: torch.Tensor
        Activation global scale, float32 device scalar.
    bias: Optional[torch.Tensor]
        Optional per-column bias, shape ``(n,)`` bf16.
    enable_pdl: Optional[bool]
        Whether to launch with Programmatic Dependent Launch. Defaults to the device default.
    backend: Literal["cutlass", "cute-dsl", "cute-dsl-unfused", "auto"]
        Backend forwarded to smooth quantization and SVDQuant GEMM. Defaults to
        architecture-based automatic selection.

    Returns
    -------
    out: torch.Tensor
        Output tensor, shape ``(m, n)`` bf16.
    """
    quantize_backend = "cute-dsl" if backend == "cute-dsl-unfused" else backend
    xq, x_sf = nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        enable_pdl=enable_pdl,
        backend=quantize_backend,
    )
    down = torch.mm(x, l2t_smoothed)
    return mm_nvfp4_svdquant(
        xq,
        weight_fp4,
        x_sf,
        weight_sf,
        alpha,
        down,
        l1_scaled,
        bias=bias,
        enable_pdl=enable_pdl,
        backend=backend,
    )
