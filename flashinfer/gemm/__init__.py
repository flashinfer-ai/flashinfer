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
    batch_deepgemm_fp8_nt_groupwise as batch_deepgemm_fp8_nt_groupwise,
)
from .gemm_base import (
    group_deepgemm_fp8_nt_groupwise as group_deepgemm_fp8_nt_groupwise,
)
from .gemm_base import gemm_fp8_nt_blockscaled as gemm_fp8_nt_blockscaled
from .gemm_base import gemm_fp8_nt_groupwise as gemm_fp8_nt_groupwise
from .gemm_base import group_gemm_fp8_nt_groupwise as group_gemm_fp8_nt_groupwise
from .gemm_base import fp8_blockscale_gemm_sm90 as fp8_blockscale_gemm_sm90

from .routergemm_dsv3 import (
    mm_M1_16_K7168_N128 as mm_M1_16_K7168_N128,
    mm_M1_16_K7168_N256 as mm_M1_16_K7168_N256,
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

__all__ = [
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
    "batch_deepgemm_fp8_nt_groupwise",
    "group_deepgemm_fp8_nt_groupwise",
    "gemm_fp8_nt_blockscaled",
    "gemm_fp8_nt_groupwise",
    "group_gemm_fp8_nt_groupwise",
    "fp8_blockscale_gemm_sm90",
    "mm_M1_16_K7168_N128",
    "mm_M1_16_K7168_N256",
] + _cute_dsl_kernels
