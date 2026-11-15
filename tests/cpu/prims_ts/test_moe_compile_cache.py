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

import contextlib
from unittest import mock

import pytest
import torch

from flashinfer.jit import cute_dsl_core
from flashinfer.prims_ts.moe import compile_cache


@pytest.mark.cpu_only
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
    io["c_row_stride"] = io["shape"][0]
    io["cfg"] = object()
    return io


@pytest.fixture
def _sm103(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 3))


@pytest.mark.cpu_only
@pytest.mark.usefixtures("_sm103")
def test_disk_kernel_name_ignores_runtime_values():
    name = compile_cache._disk_kernel_name("cfg", "fc1", _fake_launch_io())
    # Addresses and the problem shape are runtime arguments of the kernel.
    assert name == compile_cache._disk_kernel_name(
        "cfg", "fc1", _fake_launch_io(1 << 30, shape=(9216, 64, 5120, 48, 8))
    )


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_compiled_gemm_targets_the_launch_device(monkeypatch, request):
    # The launch tensors live on device 1 (SM103) while device 0 (SM100) is
    # current: the GEMM must be keyed, named, compiled and persisted for SM103.
    capabilities = {0: (10, 0), 1: (10, 3)}
    current = [0]

    @contextlib.contextmanager
    def device(index):
        previous, current[0] = current[0], index
        try:
            yield
        finally:
            current[0] = previous

    def get_device_capability(index=None):
        return capabilities[current[0] if index is None else index]

    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)
    monkeypatch.setattr(compile_cache, "_kernel_source_sha256", lambda: "sources")

    specs = []

    class MissingSpec(cute_dsl_core.JitSpecCuteDsl):
        def try_load(self):
            return None

        def compile_and_persist(self):
            specs.append((self, current[0]))

        def load(self):
            return self

    monkeypatch.setattr(cute_dsl_core, "JitSpecCuteDsl", MissingSpec)
    compile_cache._COMPILED_GEMM_CACHE.clear()
    request.addfinalizer(compile_cache._COMPILED_GEMM_CACHE.clear)

    io = _fake_launch_io()
    io["_keepalive"] = (
        mock.NonCallableMock(spec=torch.Tensor, device=torch.device("cuda", 1)),
    )
    compile_cache.get_compiled_gemm("cfg", "fc1", io, None)

    ((key, compiled),) = compile_cache._COMPILED_GEMM_CACHE.items()
    assert key[-2:] == (1, "sm103a")
    ((spec, compile_device),) = specs
    assert compiled is spec
    assert compile_device == 1
    assert spec.module_dir_name == f"{compile_cache._DISK_MODULE_NAME}_sm103a_cute_dsl"
    assert spec.expected_meta["arch"] == "sm103a"
    assert spec.kernel_name.startswith("fc1_sm103_")
