from .gemm_base import SegmentGEMMWrapper as SegmentGEMMWrapper
from .gemm_base import bmm_bf16 as bmm_bf16
from .gemm_base import bmm_fp8 as bmm_fp8
from .gemm_base import bmm_mxfp8 as bmm_mxfp8
from .gemm_base import mm_bf16 as mm_bf16
from .gemm_base import mm_fp4 as mm_fp4
from .gemm_base import mm_fp8 as mm_fp8
from .gemm_base import mm_mxfp8 as mm_mxfp8
from .gemm_base import tgv_gemm_sm100 as tgv_gemm_sm100
from .gemm_base import group_gemm_mxfp4_nt_groupwise as group_gemm_mxfp4_nt_groupwise
from .gemm_base import (
    group_gemm_mxfp8_mxfp4_nt_groupwise as group_gemm_mxfp8_mxfp4_nt_groupwise,
)
from .gemm_base import group_gemm_nvfp4_nt_groupwise as group_gemm_nvfp4_nt_groupwise
from .gemm_base import (
    batch_deepgemm_fp8_nt_groupwise as batch_deepgemm_fp8_nt_groupwise,
)
from .gemm_base import (
    group_deepgemm_fp8_nt_groupwise as group_deepgemm_fp8_nt_groupwise,
)
from .gemm_base import gemm_fp8_nt_blockscaled as gemm_fp8_nt_blockscaled
from .gemm_base import gemm_fp8_nt_groupwise as gemm_fp8_nt_groupwise
from .gemm_base import group_gemm_fp8_nt_groupwise as group_gemm_fp8_nt_groupwise
from .gemm_base import fp8_blockscale_gemm_sm90 as fp8_blockscale_gemm_sm90

from .gemm_bf16_fp4 import (
    mm_bf16_fp4 as mm_bf16_fp4,
    prepare_bf16_fp4_weights as prepare_bf16_fp4_weights,
)

from .gemm_svdquant import (
    mm_nvfp4_svdquant as mm_nvfp4_svdquant,
    nvfp4_quantize_smooth as nvfp4_quantize_smooth,
    svdquant_linear as svdquant_linear,
)

from .routergemm import (
    mm_M1_16_K6144_N256 as mm_M1_16_K6144_N256,
    mm_M1_16_K7168_N128 as mm_M1_16_K7168_N128,
    mm_M1_16_K7168_N256 as mm_M1_16_K7168_N256,
    tinygemm_bf16 as tinygemm_bf16,
)

# Import CuTe-DSL kernels if available
_cute_dsl_kernels = []
try:
    from flashinfer.cute_dsl.utils import is_cute_dsl_available

    if is_cute_dsl_available():
        from .kernels.grouped_gemm_masked_blackwell import (
            grouped_gemm_nt_masked as grouped_gemm_nt_masked,
            Sm100BlockScaledPersistentDenseGemmKernel as Sm100BlockScaledPersistentDenseGemmKernel,
            create_scale_factor_tensor as create_scale_factor_tensor,
        )

        _cute_dsl_kernels = [
            "grouped_gemm_nt_masked",
            "Sm100BlockScaledPersistentDenseGemmKernel",
            "create_scale_factor_tensor",
        ]
except ImportError:
    pass

try:
    from flashinfer.cute_dsl.utils import is_cute_dsl_available

    if is_cute_dsl_available():
        from .kernels.dense_blockscaled_gemm_sm120_b12x import (
            Sm120B12xBlockScaledDenseGemmKernel as Sm120B12xBlockScaledDenseGemmKernel,
        )

        _cute_dsl_kernels.append("Sm120B12xBlockScaledDenseGemmKernel")
except ImportError:
    pass

# is_cuda_tile_available is always importable: flashinfer.cutile.cutile_common
# has no cuda.tile imports by design, so this never fails even when cuda-tile is
# absent. Mirrors how is_cute_dsl_available is exported unconditionally
# from flashinfer.cute_dsl.
from ..cutile import (
    is_cuda_tile_available as is_cuda_tile_available,
)

_cuda_tile_kernels = ["is_cuda_tile_available"]
try:
    if is_cuda_tile_available():
        from .kernels.cutile.bmm_bf16_cutile import (
            make_bmm_bf16_tune_cache as make_bmm_bf16_tune_cache,
        )

        _cuda_tile_kernels.append("make_bmm_bf16_tune_cache")
except ImportError:
    pass

__all__ = (
    [
        "SegmentGEMMWrapper",
        "bmm_bf16",
        "bmm_fp8",
        "bmm_mxfp8",
        "mm_bf16",
        "mm_fp4",
        "mm_fp8",
        "mm_mxfp8",
        "tgv_gemm_sm100",
        "group_gemm_mxfp4_nt_groupwise",
        "group_gemm_mxfp8_mxfp4_nt_groupwise",
        "group_gemm_nvfp4_nt_groupwise",
        "batch_deepgemm_fp8_nt_groupwise",
        "group_deepgemm_fp8_nt_groupwise",
        "gemm_fp8_nt_blockscaled",
        "gemm_fp8_nt_groupwise",
        "group_gemm_fp8_nt_groupwise",
        "fp8_blockscale_gemm_sm90",
        "mm_bf16_fp4",
        "prepare_bf16_fp4_weights",
        "mm_M1_16_K6144_N256",
        "mm_M1_16_K7168_N128",
        "mm_M1_16_K7168_N256",
        "tinygemm_bf16",
    ]
    + _cute_dsl_kernels
    + _cuda_tile_kernels
)
