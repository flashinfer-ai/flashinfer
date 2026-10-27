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


def test_compiled_gemm_cache_is_partitioned_by_device_and_arch(monkeypatch, request):
    compiled = []
    target = [(0, "sm100a")]

    def fake_compile(cfg_hash, fc1_or_fc2, io, stream, device_index):
        result = object()
        compiled.append((io, stream, result))
        return result

    monkeypatch.setattr(compile_cache, "_compile_target_key", lambda _io: target[0])
    monkeypatch.setattr(compile_cache, "_compile_or_load", fake_compile)
    compile_cache._COMPILED_GEMM_CACHE.clear()
    request.addfinalizer(compile_cache._COMPILED_GEMM_CACHE.clear)

    io = {"cfg": object()}
    first = compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream")
    assert compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream") is first

    target[0] = (1, "sm100a")
    second = compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream")
    target[0] = (1, "sm103a")
    third = compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream")

    assert first is not second
    assert second is not third
    assert len(compiled) == 3


def _fake_launch_io(address: int = 1 << 20, *, align: int = 16, shape=None) -> dict:
    """Launch IO with the argument types of a real GEMM launch."""
    import cutlass
    from cutlass.cute.runtime import make_ptr

    def ptr(dtype, offset, assumed_align=align):
        return make_ptr(
            dtype, address + offset, cutlass.AddressSpace.gmem, assumed_align
        )

    io = {
        name: ptr(dtype, 256 * k)
        for k, (name, dtype) in enumerate(
            (
                ("a_dp", cutlass.Float4E2M1FN),
                ("b_dp", cutlass.Float8E4M3FN),
                ("sfa_dp", cutlass.Float8E8M0FNU),
                ("sfb_dp", cutlass.Float8E8M0FNU),
                ("c0_dp", cutlass.Float8E4M3FN),
                ("sf_c_dp", cutlass.Float8E8M0FNU),
                ("tile_idx_dp", cutlass.Int32),
                ("route_map_dp", cutlass.Int32),
                ("mn_limit_dp", cutlass.Int32),
                ("num_non_exiting_ctas_dp", cutlass.Int32),
                ("total_num_padded_tokens_dp", cutlass.Int32),
                ("act_dp", cutlass.Float8E4M3FN),
                ("per_token_sf_a_dp", cutlass.Float32),
                ("per_token_sf_b_dp", cutlass.Float32),
                ("bias_dp", cutlass.Float32),
                ("scale_c_dp", cutlass.Float32),
                ("scale_gate_dp", cutlass.Float32),
                ("gemm1_alpha_dp", cutlass.Float32),
                ("gemm1_beta_dp", cutlass.Float32),
                ("gemm1_clamp_limit_dp", cutlass.Float32),
            )
        )
    }
    io["shape"] = shape or (4608, 1024, 5120, 48, 128)
    io["launch_early_exit_max_token_ctas"] = 96
    io["cfg"] = object()
    return io


@pytest.fixture
def _sm103(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 3))


@pytest.mark.usefixtures("_sm103")
def test_disk_kernel_name_ignores_runtime_values():
    name = compile_cache._disk_kernel_name("cfg", "fc1", _fake_launch_io())
    # Addresses and the problem shape are runtime arguments of the kernel.
    assert name == compile_cache._disk_kernel_name(
        "cfg", "fc1", _fake_launch_io(1 << 30, shape=(9216, 64, 5120, 48, 8))
    )


@pytest.mark.usefixtures("_sm103")
def test_disk_kernel_name_covers_compile_time_inputs(monkeypatch):
    name = compile_cache._disk_kernel_name("cfg", "fc1", _fake_launch_io())
    others = [
        compile_cache._disk_kernel_name("cfg2", "fc1", _fake_launch_io()),
        compile_cache._disk_kernel_name("cfg", "fc2", _fake_launch_io()),
        # A pointer's assumed alignment is part of the kernel's signature.
        compile_cache._disk_kernel_name("cfg", "fc1", _fake_launch_io(align=32)),
    ]
    io = _fake_launch_io()
    import cutlass
    from cutlass.cute.runtime import make_ptr

    io["c0_dp"] = make_ptr(
        cutlass.BFloat16, 1 << 20, cutlass.AddressSpace.gmem, assumed_align=16
    )
    others.append(compile_cache._disk_kernel_name("cfg", "fc1", io))
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_COMPILE_OPTIONS", "--opt-level 3")
    others.append(compile_cache._disk_kernel_name("cfg", "fc1", _fake_launch_io()))
    monkeypatch.delenv("FLASHINFER_PRIMS_TS_COMPILE_OPTIONS")
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "1")
    others.append(compile_cache._disk_kernel_name("cfg", "fc1", _fake_launch_io()))
    monkeypatch.delenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
    others.append(compile_cache._disk_kernel_name("cfg", "fc1", _fake_launch_io()))

    assert len({name, *others}) == len(others) + 1


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
