"""JIT module for fused gated activation and MXFP8 quantization."""

from pathlib import Path

from . import env as jit_env
from .core import JitSpec, current_compilation_context, gen_jit_spec


_SOURCE_NAMES = (
    "gated_act_mxfp8.cu",
    "gated_act_mxfp8_jit_binding.cu",
    "gated_act_mxfp8_fwd_row_launch.cu",
    "gated_act_mxfp8_fwd_col_launch.cu",
    "gated_act_mxfp8_fwd_both_launch.cu",
    "gated_act_mxfp8_bwd_row_launch.cu",
    "gated_act_mxfp8_bwd_col_launch.cu",
    "gated_act_mxfp8_bwd_both_launch.cu",
)


def _source_dir() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / "gated_act_mxfp8"
    if packaged.is_dir():
        return packaged

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "gated_act_mxfp8"
    if checkout.is_dir():
        return checkout

    raise FileNotFoundError("gated_act_mxfp8 CUDA sources are not installed")


def gen_gated_act_mxfp8_module() -> JitSpec:
    source_dir = _source_dir()
    return gen_jit_spec(
        "gated_act_mxfp8_sm10x",
        [source_dir / name for name in _SOURCE_NAMES],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[10],
            map_sm107_to_100f=False,
        ),
        extra_include_paths=[
            str(source_dir.parent),
            str(source_dir.parent.parent / "include"),
        ],
    )


__all__ = ["gen_gated_act_mxfp8_module"]
