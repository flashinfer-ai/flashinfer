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

import pytest
import torch

from flashinfer.jit import core
from flashinfer.jit import env as jit_env
from flashinfer.jit.gemm import gen_tgv_gemm_sm10x_module


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("use_sm_100f", "gencode"),
    [
        (False, "-gencode=arch=compute_100a,code=sm_100a"),
        (True, "-gencode=arch=compute_100f,code=sm_100f"),
    ],
    ids=["sm100a", "sm100f"],
)
def test_every_tgv_gemm_target_enables_gdc_and_relaxed_constexpr(
    monkeypatch, tmp_path, use_sm_100f, gencode
):
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)

    flags = gen_tgv_gemm_sm10x_module(
        torch.bfloat16, use_sm_100f=use_sm_100f
    ).extra_cuda_cflags

    # Without the GDC define, CUTLASS compiles the kernel's PDL wait to a no-op.
    assert "-DCUTLASS_ENABLE_GDC_FOR_SM100=1" in flags
    assert "--expt-relaxed-constexpr" in flags
    assert [flag for flag in flags if flag.startswith("-gencode=")] == [gencode]
