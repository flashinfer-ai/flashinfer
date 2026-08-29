"""JIT generator for ReplaySSM prefix-state materialization."""

import os
import jinja2
import torch
from ...compilation_context import CompilationContext
from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec
from ..utils import write_if_different

_DTYPE = {
    torch.float16: "half",
    torch.bfloat16: "nv_bfloat16",
    torch.float32: "float",
    torch.int8: "int8_t",
    torch.float8_e4m3fn: "__nv_fp8_e4m3",
}
_NAME = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.int8: "i8",
    torch.float8_e4m3fn: "e4m3",
}


def gen_replayssm_materialize_module(
    state_dtype,
    input_dtype,
    matrixA_dtype,
    dim,
    dstate,
    heads_per_group,
    max_window,
    philox_rounds=0,
) -> JitSpec:
    uri = (
        f"replayssm_materialize_s_{_NAME[state_dtype]}_i_{_NAME[input_dtype]}"
        f"_a_{_NAME[matrixA_dtype]}_d_{dim}_ds_{dstate}_hpg_{heads_per_group}"
        f"_mw_{max_window}_pr_{philox_rounds}"
    )
    directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(directory, exist_ok=True)
    with open(
        jit_env.FLASHINFER_CSRC_DIR / "replayssm_materialize_customize_config.jinja"
    ) as f:
        config = jinja2.Template(f.read()).render(
            state_dtype=_DTYPE[state_dtype],
            input_dtype=_DTYPE[input_dtype],
            matrixA_dtype=_DTYPE[matrixA_dtype],
            dim=dim,
            dstate=dstate,
            heads_per_group=heads_per_group,
            max_window=max_window,
            philox_rounds=philox_rounds,
        )
    write_if_different(directory / "replayssm_materialize_config.inc", config)
    sources = []
    for name in ("replayssm_materialize.cu", "replayssm_materialize_jit_binding.cu"):
        dst = directory / name
        write_if_different(dst, (jit_env.FLASHINFER_CSRC_DIR / name).read_text())
        sources.append(dst)
    flags = CompilationContext().get_nvcc_flags_list(
        supported_major_versions=[8, 9, 10, 11, 12]
    )
    return gen_jit_spec(uri, sources, extra_cuda_cflags=flags)
