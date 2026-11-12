"""JIT generator for ReplaySSM prefix-state materialization.

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

_INPUT_DTYPE = torch.bfloat16


def gen_replayssm_materialize_module(
    state_dtype,
    input_dtype,
    matrixA_dtype,
    dim,
    dstate,
    heads_per_group,
    max_window,
    philox_rounds=0,
    sr_rng="philox",
) -> JitSpec:
    if input_dtype != _INPUT_DTYPE:
        raise ValueError(
            "ReplaySSM materialization requires input_dtype=torch.bfloat16: "
            "the replay x and B caches are BF16 MMA operands"
        )
    if sr_rng not in ("philox", "shared_seed"):
        raise ValueError("sr_rng must be 'philox' or 'shared_seed'")
    if sr_rng == "shared_seed":
        if philox_rounds < 0:
            raise ValueError("shared_seed requires nonnegative philox_rounds")
        if state_dtype not in (torch.float16, torch.float8_e4m3fn, torch.int8):
            raise ValueError("shared_seed requires FP16, FP8 E4M3 or INT8 state")
        philox_rounds = philox_rounds or 10
    uri = (
        f"replayssm_materialize_s_{_NAME[state_dtype]}_i_{_NAME[input_dtype]}"
        f"_a_{_NAME[matrixA_dtype]}_d_{dim}_ds_{dstate}_hpg_{heads_per_group}"
        f"_mw_{max_window}_pr_{philox_rounds}"
    )
    if sr_rng == "shared_seed":
        uri += "_sr_shared_seed"
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
            shared_seed=sr_rng == "shared_seed",
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
    if sr_rng == "shared_seed":
        flags += ["-DFLASHINFER_MAMBA_SR_SHARED_SEED=1"]
    return gen_jit_spec(uri, sources, extra_cuda_cflags=flags)
