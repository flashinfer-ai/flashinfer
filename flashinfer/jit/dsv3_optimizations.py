from .core import JitSpec, gen_jit_spec
from . import env as jit_env


def gen_concat_mla_module() -> JitSpec:
    """Generate JIT spec for concat_mla kernel.

    This kernel efficiently concatenates CKV and KPE tensors for MLA prefill attention
    """
    return gen_jit_spec(
        "concat_mla",
        [
            jit_env.FLASHINFER_CSRC_DIR / "concat_mla.cu",
        ],
    )


def gen_dsv3_router_gemm_module(
    num_experts: int = 256,
    hidden_dim: int = 7168,
    out_float: bool = True,
) -> JitSpec:
    """JIT spec for one router-GEMM specialization.

    The kernel keeps the expert count, the hidden dim and the output dtype as
    compile-time constants (they drive the grid dimension, the unrolled K loop
    and the store type respectively), so each combination compiles to its own
    small module. Defaults are the DeepSeek-V3 router shape.
    """
    return gen_jit_spec(
        f"dsv3_router_gemm_n{num_experts}_k{hidden_dim}_"
        + ("f32" if out_float else "bf16"),
        [
            jit_env.FLASHINFER_CSRC_DIR / "dsv3_router_gemm.cu",
        ],
        extra_cuda_cflags=[
            f"-DROUTER_GEMM_NUM_EXPERTS={num_experts}",
            f"-DROUTER_GEMM_HIDDEN_DIM={hidden_dim}",
            f"-DROUTER_GEMM_OUT_FLOAT={1 if out_float else 0}",
        ],
    )


def gen_dsv3_fused_routing_module() -> JitSpec:
    return gen_jit_spec(
        "dsv3_fused_routing",
        [
            jit_env.FLASHINFER_CSRC_DIR / "fused_moe/noAuxTcKernels.cu",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/memoryUtils.cu",
        ],
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "cutlass_extensions"
            / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "cutlass_kernels"
            / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "cutlass_kernels",
        ],
    )
