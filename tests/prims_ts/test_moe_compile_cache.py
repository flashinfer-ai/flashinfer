# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flashinfer.jit import env as jit_env
from flashinfer.prims_ts.batched_gemm import batched_gemm_run
from flashinfer.prims_ts.moe import compile_cache


def _prims_ts_gpu_available() -> bool:
    from flashinfer.prims_ts.utils import is_prims_ts_available
    from flashinfer.utils import get_compute_capability

    if not torch.cuda.is_available() or not is_prims_ts_available():
        return False
    return get_compute_capability(torch.device("cuda")) in ((10, 0), (10, 3))


def _run_mxfp4_mxfp8_moe() -> torch.Tensor:
    """One small Prims-TS MXFP4 x MXFP8 MoE forward with fixed inputs."""
    from flashinfer.fused_moe import MoELayer, MoEWeightPack, PrimsTsConfig
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEActivationPack,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        SwiGLU,
    )

    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    num_experts, hidden, intermediate, tokens, top_k = 4, 256, 256, 8, 2
    quant = QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts, 2 * intermediate, hidden, device=device, dtype=torch.bfloat16
    )
    w2 = torch.randn(
        num_experts, hidden, intermediate, device=device, dtype=torch.bfloat16
    )
    weights, ids = torch.topk(
        torch.softmax(torch.randn(tokens, num_experts, device=device), dim=-1),
        top_k,
        dim=-1,
    )
    x_q, x_scale = PrimsTsConfig.prepare_activations(x, quant=quant)
    pack = MoEWeightPack()
    pack.prepare_for(
        "prims_ts",
        PrimsTsConfig.prepare_weights(
            w1,
            w2,
            quant=quant,
            num_local_experts=num_experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            activation=SwiGLU(),
            device=device,
        ),
    )
    layer = MoELayer(
        MoEConfig(
            routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
            quant=quant,
            experts=ExpertConfig(intermediate_size=intermediate),
            activation=SwiGLU(),
            backend=BackendOptions((PrimsTsConfig(),)),
            execution=ExecutionConfig(tune_max_num_tokens=tokens),
        )
    )
    act = MoEActivationPack(x_q, x_scale, ids.to(torch.int32), weights)
    out = layer(act, pack)
    torch.cuda.synchronize()
    return out


@pytest.mark.skipif(
    not _prims_ts_gpu_available(),
    reason="Prims-TS MoE requires SM100/SM103 and nvidia-cutlass-dsl",
)
def test_compiled_gemms_are_reused_from_disk(monkeypatch, request, tmp_path):
    compiles = []
    compile_for_launch = batched_gemm_run._compile_for_launch

    def counting_compile(io, stream):
        compiles.append(io["cfg"])
        return compile_for_launch(io, stream)

    monkeypatch.setattr(batched_gemm_run, "_compile_for_launch", counting_compile)
    request.addfinalizer(compile_cache._COMPILED_GEMM_CACHE.clear)

    # Reference: kernels compiled in-process, nothing read from or written to disk.
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    compile_cache._COMPILED_GEMM_CACHE.clear()
    reference = _run_mxfp4_mxfp8_moe()
    assert len(compiles) == 2
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE")

    # An empty JIT directory: FC1 and FC2 compile once and persist.
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path)
    compile_cache._COMPILED_GEMM_CACHE.clear()
    first = _run_mxfp4_mxfp8_moe()
    assert len(compiles) == 4
    objects = list(tmp_path.glob(f"{compile_cache._DISK_MODULE_NAME}_*/*.o"))
    assert len(objects) == 2

    # A process with an empty in-process cache loads both from disk.
    compile_cache._COMPILED_GEMM_CACHE.clear()
    second = _run_mxfp4_mxfp8_moe()
    assert len(compiles) == 4

    assert torch.equal(first, reference)
    assert torch.equal(second, reference)
