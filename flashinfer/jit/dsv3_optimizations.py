from pathlib import Path

from .core import JitSpec, gen_jit_spec
from . import env as jit_env


def _get_concat_mla_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR
    if (installed / "concat_mla.cu").is_file():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc"
    if (checkout / "concat_mla.cu").is_file():
        return checkout
    raise FileNotFoundError(
        "FlashInfer concat MLA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_concat_mla_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


def gen_concat_mla_module() -> JitSpec:
    """Generate JIT spec for concat_mla kernel.

    This kernel efficiently concatenates CKV and KPE tensors for MLA prefill attention
    """
    csrc_dir = _get_concat_mla_csrc_dir()
    return gen_jit_spec(
        "concat_mla",
        [
            csrc_dir / "concat_mla.cu",
        ],
        extra_include_paths=[csrc_dir, _get_concat_mla_include_dir()],
    )


def gen_dsv3_router_gemm_module() -> JitSpec:
    return gen_jit_spec(
        "dsv3_router_gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "dsv3_router_gemm.cu",
        ],
    )


def gen_dsv3_fused_routing_module(backend: str = "default") -> JitSpec:
    if backend not in ("default", "cake"):
        raise ValueError(f"Unsupported fused routing backend: {backend}")
    return gen_jit_spec(
        "dsv3_fused_routing" if backend == "default" else "dsv3_fused_routing_cake",
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
        extra_cuda_cflags=["-DFLASHINFER_CAKE_BACKEND", "--use_fast_math"]
        if backend == "cake"
        else None,
    )
