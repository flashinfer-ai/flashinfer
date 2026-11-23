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
import gc
from types import SimpleNamespace
from unittest import mock
import weakref

import pytest
import torch

from flashinfer.jit import cute_dsl_core
from flashinfer.prims_ts.moe import compile_cache


@pytest.mark.cpu_only
def test_compiled_gemm_cache_is_partitioned_by_device_and_arch(monkeypatch, request):
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_REENCODE_TMA", "0")
    compiled = []
    target = [(0, "sm100a")]

    def fake_compile(cfg_hash, fc1_or_fc2, io, stream, device_index):
        result = object()
        compiled.append((io, stream, result))
        return compile_cache._CompiledGemm(result)

    monkeypatch.setattr(compile_cache, "_compile_target_key", lambda _io: target[0])
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run

    monkeypatch.setattr(compile_cache, "_compile_or_load", fake_compile)
    monkeypatch.setattr(
        batched_gemm_run, "_prepare_tma_descriptors", lambda io, stream, **kw: None
    )
    compile_cache._COMPILED_GEMM_CACHE.clear()
    request.addfinalizer(compile_cache._COMPILED_GEMM_CACHE.clear)

    io = {"cfg": object()}
    first = compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream")
    assert compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream") is first

    target[0] = (1, "sm100a")
    second = compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream")
    target[0] = (1, "sm103a")
    third = compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream")

    monkeypatch.setenv("FLASHINFER_PRIMS_TS_REENCODE_TMA", "1")
    reencoded = compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream")
    assert reencoded is not third
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_REENCODE_TMA", "0")
    assert compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream") is third

    assert first is not second
    assert second is not third
    assert len(compiled) == 4


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
    io["cfg"] = SimpleNamespace(fused_finalize_top_k=0)
    io["tma_desc_storage_dp"] = make_ptr(
        cutlass.Uint8,
        address + 8192,
        cutlass.AddressSpace.generic,
        assumed_align=64,
    )
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
    io = _fake_launch_io()
    io["tma_desc_storage_dp"] = None
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
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run

    monkeypatch.setattr(
        batched_gemm_run, "_prepare_tma_descriptors", lambda io, stream: None
    )
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
    assert compiled.kernel is spec
    assert compile_device == 1
    assert spec.module_dir_name == f"{compile_cache._DISK_MODULE_NAME}_sm103a_cute_dsl"
    assert spec.expected_meta["arch"] == "sm103a"
    assert spec.kernel_name.startswith("fc1_sm103_")


@pytest.mark.cpu_only
def test_compiled_gemm_cache_prepares_new_workspace_on_cache_hit(monkeypatch, request):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run

    compiled = []
    prepared = []
    current = [0]
    preparer = object()

    @contextlib.contextmanager
    def device(index):
        previous, current[0] = current[0], index
        try:
            yield
        finally:
            current[0] = previous

    def prepare(io, stream, *, prepare=None):
        assert current[0] == 1
        assert prepare is None or prepare is preparer
        io["descriptor_binding"] = io["binding"]
        prepared.append((io, stream))
        return preparer

    def compile_for_launch(cfg_hash, fc1_or_fc2, io, stream, device_index):
        with device(device_index):
            prepare(io, stream)
        fn = object()
        compiled.append(fn)
        return compile_cache._CompiledGemm(fn, preparer)

    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(compile_cache, "_compile_target_key", lambda io: (1, "sm107a"))
    monkeypatch.setattr(compile_cache, "_compile_or_load", compile_for_launch)
    monkeypatch.setattr(batched_gemm_run, "_prepare_tma_descriptors", prepare)
    compile_cache.clear_compiled_gemm_cache()
    request.addfinalizer(compile_cache.clear_compiled_gemm_cache)

    first_io = {"binding": "first shard"}
    second_io = {"binding": "second shard"}
    first = compile_cache.get_compiled_gemm("cfg", "fc1", first_io, "first stream")
    second = compile_cache.get_compiled_gemm("cfg", "fc1", second_io, "second stream")
    assert first is second
    assert len(compiled) == 1
    assert second_io["descriptor_binding"] == "second shard"
    assert prepared == [(first_io, "first stream"), (second_io, "second stream")]

    second_io["binding"] = "rebound shard"
    assert (
        compile_cache.get_compiled_gemm("cfg", "fc1", second_io, "second stream")
        is first
    )
    assert second_io["descriptor_binding"] == "rebound shard"
    assert current[0] == 0


@pytest.mark.cpu_only
@pytest.mark.parametrize("disk_hit", [False, True])
def test_disk_cache_prepares_each_workspace_before_keying(monkeypatch, disk_hit):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run

    current = [0]
    prepared = []
    compiled = []
    kernel = object()

    @contextlib.contextmanager
    def device(index):
        previous, current[0] = current[0], index
        try:
            yield
        finally:
            current[0] = previous

    def prepare(io, stream):
        assert current[0] == 1
        io["descriptor_binding"] = io["binding"]
        prepared.append((io["binding"], stream))

    def name(cfg_hash, fc1_or_fc2, io):
        assert io["descriptor_binding"] == io["binding"]
        return "gemm"

    class Spec:
        def __init__(self, module, name, compile_fn, source, **kwargs):
            self.compile_fn = compile_fn

        def try_load(self):
            return kernel if disk_hit else None

        def compile_and_persist(self):
            compiled.append(self.compile_fn())

        def load(self):
            return kernel

    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(compile_cache, "_kernel_source_sha256", lambda: "sources")
    monkeypatch.setattr(compile_cache, "_disk_kernel_name", name)
    monkeypatch.setattr(cute_dsl_core, "cute_dsl_cache_disabled", lambda: False)
    monkeypatch.setattr(cute_dsl_core, "JitSpecCuteDsl", Spec)
    monkeypatch.setattr(batched_gemm_run, "_prepare_tma_descriptors", prepare)
    monkeypatch.setattr(
        batched_gemm_run, "_compile_for_launch", lambda io, stream: kernel
    )

    # Fresh IO objects simulate separate processes loading the same artifact.
    for binding in ("first shard", "second shard"):
        io = {"binding": binding}
        assert (
            compile_cache._compile_or_load("cfg", "fc1", io, "stream", 1).kernel
            is kernel
        )
        assert io["descriptor_binding"] == binding
    assert current[0] == 0
    assert prepared == [("first shard", "stream"), ("second shard", "stream")]
    assert len(compiled) == (0 if disk_hit else 2)


@pytest.mark.cpu_only
@pytest.mark.parametrize("sm_arch", [100, 103, 107])
def test_descriptor_preparation_reuses_code_without_retaining_bindings(
    monkeypatch, request, sm_arch
):
    from cutlass.cutlass_dsl import BaseDSL
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import make_config

    monkeypatch.delenv("FLASHINFER_PRIMS_TS_REENCODE_TMA", raising=False)
    monkeypatch.setattr(BaseDSL, "enable_pyir", contextlib.nullcontext)
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(
        compile_cache, "_compile_target_key", lambda io: (0, f"sm{sm_arch}a")
    )
    compile_cache.clear_compiled_gemm_cache()
    request.addfinalizer(compile_cache.clear_compiled_gemm_cache)
    compilations, bindings, owners = [], [], []
    fail = False

    class Prepare:
        def __call__(self, args):
            if fail:
                raise RuntimeError("injected descriptor preparation failure")
            bindings.append((args[0]._pointer, args[4]._pointer, args[25], args[27]))

    def compile_prepare(fn, args, cfg, **kwargs):
        assert fn is runner.prepare_gemm_descriptors
        prepared = Prepare()
        compilations.append(weakref.ref(prepared))
        return prepared

    def compile_gemm(cfg_hash, stage, io, stream, device):
        return compile_cache._CompiledGemm(
            object(), runner._prepare_tma_descriptors(io, stream)
        )

    monkeypatch.setattr(runner.cute, "compile", compile_prepare)
    monkeypatch.setattr(compile_cache, "_compile_or_load", compile_gemm)
    cfg = make_config(sm_arch=sm_arch)
    for index in range(4):
        io = _fake_launch_io(
            (index + 1) << 20, shape=(256, 512 * (index + 1), 256, 2, 128)
        )
        io.update(cfg=cfg, c_row_stride=256 + 32 * index)
        io["_keepalive"] = (torch.empty(128),)
        compiled = compile_cache.get_compiled_gemm("cfg", "fc1", io, None)
        assert bindings[-1] == (
            io["a_dp"]._pointer,
            io["c0_dp"]._pointer,
            io["shape"],
            io["c_row_stride"],
        )
        assert len(compilations) == 1
        assert len(compile_cache._COMPILED_GEMM_CACHE) == 1
        owners.extend(
            weakref.ref(value)
            for value in (*io["_keepalive"], io["_tma_descriptor_storage"])
        )
        if index == 0:
            first = compiled
        else:
            assert compiled is first
        if index == 3:
            cached = tuple(
                io[name]
                for name in (
                    "tma_desc_storage_dp",
                    "_tma_descriptor_storage",
                    "_tma_descriptor_key",
                )
            )
            io["c0_dp"] = _fake_launch_io(1 << 28)["c0_dp"]
            fail = True
            with pytest.raises(RuntimeError, match="injected"):
                compile_cache.get_compiled_gemm("cfg", "fc1", io, None)
            assert all(
                io[name] is value
                for name, value in zip(
                    (
                        "tma_desc_storage_dp",
                        "_tma_descriptor_storage",
                        "_tma_descriptor_key",
                    ),
                    cached,
                    strict=True,
                )
            )
            fail = False
            compile_cache.get_compiled_gemm("cfg", "fc1", io, None)
            assert bindings[-1][1] == io["c0_dp"]._pointer
            del cached
        del io
        gc.collect()
        assert all(owner() is None for owner in owners)
    assert len(compilations) == 1
    compile_cache.clear_compiled_gemm_cache()
    gc.collect()
    assert compilations[0]() is None
