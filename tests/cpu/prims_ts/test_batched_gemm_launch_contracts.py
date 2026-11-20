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
from dataclasses import replace
import gc
import weakref

import pytest

from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import make_config
from tests.cpu.prims_ts.test_moe_compile_cache import _fake_launch_io

pytestmark = pytest.mark.cpu_only


def _io(**options):
    io = _fake_launch_io()
    io["cfg"] = make_config(sm_arch=107, **options)
    return io


@pytest.mark.parametrize(
    "builder", [runner._compile_arg_tuple, runner._launch_arg_tuple]
)
@pytest.mark.parametrize("missing", ["finalize_scale_dp", "finalize_route_dp"])
@pytest.mark.parametrize("null", [False, True])
def test_native_finalize_requires_both_buffers(builder, missing, null):
    io = _io(fuse_moe_finalize=1)
    io.update(finalize_scale_dp=object(), finalize_route_dp=object())
    if null:
        io[missing] = None
    else:
        del io[missing]
    with pytest.raises(ValueError, match=missing):
        builder(io, None)


@pytest.mark.parametrize(
    "builder", [runner._compile_arg_tuple, runner._launch_arg_tuple]
)
@pytest.mark.parametrize("finalize", [0, 1])
def test_finalize_arguments_preserve_explicit_buffers_and_dummy_fallbacks(
    builder, finalize
):
    io = _io(fuse_moe_finalize=finalize)
    if not finalize:
        args = builder(io, None)
        assert args[23] is io["scale_c_dp"] and args[24] is io["route_map_dp"]
    io.update(finalize_scale_dp=object(), finalize_route_dp=object())
    args = builder(io, None)
    assert args[23] is io["finalize_scale_dp"]
    assert args[24] is io["finalize_route_dp"]


@pytest.mark.parametrize("builder", [runner._build_launch_io, runner.reference_check])
def test_standalone_runner_rejects_native_finalize(builder):
    with pytest.raises(ValueError, match="zeroed token output"):
        builder(
            num_experts=1,
            num_tokens=8,
            top_k=1,
            sm_arch=107,
            fuse_moe_finalize=1,
            input_device="cpu",
        )


@pytest.mark.parametrize(
    "builder", [runner._compile_arg_tuple, runner._launch_arg_tuple]
)
@pytest.mark.parametrize("experts", [-1, 0, 1, 2])
def test_single_expert_launch_contract(builder, experts):
    io = _io(dense_single_expert=1)
    io["shape"] = (*io["shape"][:3], experts, io["shape"][4])
    if experts != 1:
        with pytest.raises(ValueError, match="num_experts_dim=1"):
            builder(io, None)
    else:
        builder(io, None)
    io["cfg"] = replace(io["cfg"], dense_single_expert=0)
    if experts > 0:
        builder(io, None)


@pytest.mark.parametrize("failure_stage", ["compile", "prepare"])
def test_descriptor_rebinding_failure_preserves_cached_binding(
    monkeypatch, failure_stage
):
    from cutlass.cutlass_dsl import BaseDSL

    monkeypatch.delenv("FLASHINFER_PRIMS_TS_REENCODE_TMA", raising=False)
    monkeypatch.setattr(BaseDSL, "enable_pyir", contextlib.nullcontext)
    io = _io()
    output_a, output_b = io["c0_dp"], _fake_launch_io(1 << 24)["c0_dp"]
    cached_names = (
        "tma_desc_storage_dp",
        "_tma_descriptor_storage",
        "_tma_descriptor_key",
    )
    fail = False
    preparations = []
    storage_refs = []
    allocate = runner.ctypes.create_string_buffer

    def storage(*args):
        result = allocate(*args)
        storage_refs.append(weakref.ref(result))
        return result

    def compile_prepare(fn, args, cfg, **kwargs):
        preparations.append(args)
        assert args[-1] is not io["tma_desc_storage_dp"]
        if fail and failure_stage == "compile":
            raise RuntimeError("injected compile failure")

        def prepare(args):
            if fail:
                raise RuntimeError("injected prepare failure")

        return prepare

    monkeypatch.setattr(runner.ctypes, "create_string_buffer", storage)
    monkeypatch.setattr(runner.cute, "compile", compile_prepare)
    runner._prepare_tma_descriptors(io, None)
    cached = tuple(io[name] for name in cached_names)
    io["c0_dp"] = output_b
    fail = True
    with pytest.raises(RuntimeError, match="injected"):
        runner._prepare_tma_descriptors(io, None)
    assert all(
        io[name] is value for name, value in zip(cached_names, cached, strict=True)
    )
    gc.collect()
    assert storage_refs[-1]() is None

    io["c0_dp"] = output_a
    runner._prepare_tma_descriptors(io, None)
    assert len(preparations) == 2  # A is still cached after B failed.
    io["c0_dp"] = output_b
    fail = False
    runner._prepare_tma_descriptors(io, None)
    assert len(preparations) == 3
    assert all(
        io[name] is not value for name, value in zip(cached_names, cached, strict=True)
    )
    assert io["tma_desc_storage_dp"] is preparations[-1][-1]
