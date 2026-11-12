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

"""``enable_pdl`` plumbing of the dense PrimsTS GEMM, checked without a GPU.

The public entry points run on CPU tensors up to the launch: CUDA validation,
the compute capability, the occupancy query and the launch itself are stubbed,
and everything between them is the code a GPU call runs.
"""

import ast
import inspect
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Optional

import pytest
import torch

import flashinfer.utils as flashinfer_utils
from flashinfer.autotuner import AutoTuner
from flashinfer.prims_ts.gemm import (
    PreparedFp4Linear,
    api,
    fp4_linear,
    fp4_linear_swiglu,
    fp4_qkv_qknorm_rope,
    fp8_linear,
    fp8_linear_swiglu,
    fp8_qkv_qknorm_rope,
    prepare_fp4_linear,
    prepare_fp4_linear_swiglu,
    prepare_fp4_qkv_qknorm_rope,
    runner,
)
from flashinfer.prims_ts.gemm.config import PrimsTsGemmConfig
from flashinfer.prims_ts.gemm.runner import GemmIdentity, PrimsTsGemmRunner
from flashinfer.prims_ts.gemm.support import nvfp4_128x4_numel
from flashinfer.prims_ts.gemm.tactics import config_from_tactic, fallback_config

pytestmark = pytest.mark.cpu_only

# K is one NVFP4 TMA box; N holds one Q, K and V head of 128.
M, N, K = 128, 384, 256
_HEADS = dict(num_q_heads=1, num_kv_heads=1, head_dim=128)


def _fp8_operands():
    return (
        torch.zeros((M, K), dtype=torch.float8_e4m3fn),
        torch.zeros((N, K), dtype=torch.float8_e4m3fn),
        torch.ones(M),
        torch.ones(N),
    )


def _fp4_activation():
    return (
        torch.zeros((M, K // 2), dtype=torch.uint8),
        torch.zeros((nvfp4_128x4_numel(M, K),), dtype=torch.uint8),
    )


def _fp4_weight():
    return (
        torch.zeros((N, K // 2), dtype=torch.uint8),
        torch.zeros((nvfp4_128x4_numel(N, K),), dtype=torch.uint8),
    )


def _norms():
    return torch.ones(128, dtype=torch.bfloat16), torch.ones(128, dtype=torch.bfloat16)


def _rope():
    return torch.zeros((M, 128)), torch.zeros(M, dtype=torch.int64)


def _fp4_one_shot(fn, *extra, **kwargs):
    a, sfa = _fp4_activation()
    weight, sfw = _fp4_weight()
    return fn(a, sfa, 1.0, weight, sfw, 1.0, *extra, skip_check=True, **kwargs)


ONE_SHOT = {
    "fp8_linear": lambda **kw: fp8_linear(*_fp8_operands(), skip_check=True, **kw),
    "fp8_linear_swiglu": lambda **kw: fp8_linear_swiglu(
        *_fp8_operands(), skip_check=True, **kw
    ),
    "fp8_qkv_qknorm_rope": lambda **kw: fp8_qkv_qknorm_rope(
        *_fp8_operands(), *_norms(), *_rope(), **_HEADS, skip_check=True, **kw
    ),
    "fp4_linear": lambda **kw: _fp4_one_shot(fp4_linear, **kw),
    "fp4_linear_swiglu": lambda **kw: _fp4_one_shot(fp4_linear_swiglu, **kw),
    "fp4_qkv_qknorm_rope": lambda **kw: _fp4_one_shot(
        fp4_qkv_qknorm_rope, *_norms(), *_rope(), **_HEADS, **kw
    ),
}

PREPARE = {
    "prepare_fp4_linear": lambda **kw: prepare_fp4_linear(
        *_fp4_weight(), 1.0, 1.0, **kw
    ),
    "prepare_fp4_linear_swiglu": lambda **kw: prepare_fp4_linear_swiglu(
        *_fp4_weight(), 1.0, 1.0, **kw
    ),
    "prepare_fp4_qkv_qknorm_rope": lambda **kw: prepare_fp4_qkv_qknorm_rope(
        *_fp4_weight(), 1.0, 1.0, *_norms(), **_HEADS, **kw
    ),
}

PUBLIC = {
    "fp8_linear": fp8_linear,
    "fp8_linear_swiglu": fp8_linear_swiglu,
    "fp8_qkv_qknorm_rope": fp8_qkv_qknorm_rope,
    "fp4_linear": fp4_linear,
    "fp4_linear_swiglu": fp4_linear_swiglu,
    "fp4_qkv_qknorm_rope": fp4_qkv_qknorm_rope,
    "prepare_fp4_linear": prepare_fp4_linear,
    "prepare_fp4_linear_swiglu": prepare_fp4_linear_swiglu,
    "prepare_fp4_qkv_qknorm_rope": prepare_fp4_qkv_qknorm_rope,
    "PreparedFp4Linear": PreparedFp4Linear.__init__,
}


@pytest.fixture
def launches(monkeypatch):
    """Stub the GPU-only edges; record what reaches the launch."""
    recorded = []
    monkeypatch.setattr(api, "_validate_cuda_tensors", lambda *_: torch.device("cpu"))
    monkeypatch.setattr(api, "get_compute_capability", lambda _device: (10, 0))
    monkeypatch.setattr(api, "_max_active_clusters", lambda *_: 1)
    monkeypatch.setattr(
        runner, "launch_dense_gemm", lambda op_name, **kw: recorded.append(kw)
    )
    monkeypatch.setattr(
        api, "_launch", lambda config, *_, **__: recorded.append({"config": config})
    )

    def compile_(config, module, values):
        recorded.append({"config": config, "module": module})
        return lambda *_: None

    monkeypatch.setattr(api, "_compile", compile_)
    return recorded


def _device_supports_pdl(monkeypatch, supported):
    monkeypatch.setattr(flashinfer_utils, "device_support_pdl", lambda _: supported)


def _base_config(**overrides):
    return replace(
        PrimsTsGemmConfig(100, "nvfp4_e2m1", "bf16", "linear", False, None, None),
        tile_k=256,
        **overrides,
    )


@pytest.mark.parametrize("name", PUBLIC)
def test_enable_pdl_is_the_last_keyword(name):
    parameters = list(
        inspect.signature(PUBLIC[name], eval_str=True).parameters.values()
    )
    last = parameters[-1]
    assert last.name == "enable_pdl"
    assert last.kind is inspect.Parameter.KEYWORD_ONLY
    assert last.default is None
    assert last.annotation == Optional[bool]


def test_prepared_call_takes_no_enable_pdl():
    # The flag is compiled into the prepared kernel; calls cannot change it.
    assert "enable_pdl" not in inspect.signature(PreparedFp4Linear.__call__).parameters


@pytest.mark.parametrize("name", [*ONE_SHOT, *PREPARE])
@pytest.mark.parametrize("value", (0, 1, "true"))
def test_non_bool_enable_pdl_raises(launches, name, value):
    call = {**ONE_SHOT, **PREPARE}[name]
    with pytest.raises(TypeError, match="enable_pdl must be a bool or None"):
        call(enable_pdl=value)
    assert launches == []


@pytest.mark.parametrize("supported", (False, True))
@pytest.mark.parametrize("enable_pdl", (None, False, True))
@pytest.mark.parametrize("name", ONE_SHOT)
def test_one_shot_launches_the_resolved_flag(
    launches, monkeypatch, name, enable_pdl, supported
):
    _device_supports_pdl(monkeypatch, supported)
    ONE_SHOT[name](enable_pdl=enable_pdl)
    expected = supported if enable_pdl is None else enable_pdl
    assert launches[-1]["enable_pdl"] is expected


@pytest.mark.parametrize("supported", (False, True))
@pytest.mark.parametrize("enable_pdl", (None, False, True))
@pytest.mark.parametrize("name", PREPARE)
def test_prepared_projection_keeps_the_flag_resolved_at_prepare(
    launches, monkeypatch, name, enable_pdl, supported
):
    _device_supports_pdl(monkeypatch, supported)
    prepared = PREPARE[name](enable_pdl=enable_pdl)
    expected = supported if enable_pdl is None else enable_pdl
    assert prepared.config.enable_pdl is expected
    assert prepared._module.enable_pdl is expected
    # Device support may change later; the prepared choice does not.
    _device_supports_pdl(monkeypatch, not supported)
    rope = (
        dict(zip(("cos_sin", "positions"), _rope(), strict=True))
        if "qkv" in name
        else {}
    )
    prepared(*_fp4_activation(), **rope)
    assert launches[-1]["enable_pdl"] is expected


@pytest.mark.parametrize("value", (False, True))
def test_explicit_config_carries_the_choice(launches, monkeypatch, value):
    # The config is the whole compile-time decision: None keeps its flag, even
    # where the device would resolve None the other way, and a bool must match.
    _device_supports_pdl(monkeypatch, not value)
    config = fallback_config(
        arch=100,
        operand_format="fp8_e4m3",
        output_format="bf16",
        epilogue="linear",
        has_bias=False,
        head_dim=None,
        is_neox=None,
        has_qkv_scale=False,
        enable_pdl=value,
    )
    for enable_pdl in (None, value):
        ONE_SHOT["fp8_linear"](config=config, enable_pdl=enable_pdl)
        assert launches[-1]["config"] is config
    with pytest.raises(ValueError, match="does not match config.enable_pdl"):
        ONE_SHOT["fp8_linear"](config=config, enable_pdl=not value)

    fp4_config = _base_config(enable_pdl=value)
    prepared = PREPARE["prepare_fp4_linear"](config=fp4_config)
    assert prepared.config is fp4_config
    prepared(*_fp4_activation())
    assert launches[-1]["config"] is fp4_config
    with pytest.raises(ValueError, match="does not match config.enable_pdl"):
        PREPARE["prepare_fp4_linear"](config=fp4_config, enable_pdl=not value)


def test_pdl_modes_are_distinct_kernels_with_one_tactic_list():
    identity = dict(
        arch=103,
        operand_format="nvfp4_e2m1",
        output_format="bf16",
        epilogue="linear",
        has_bias=False,
        head_dim=None,
        is_neox=None,
        has_qkv_scale=False,
    )
    on, off = (
        PrimsTsGemmRunner("prims_ts_fp4_linear", GemmIdentity(**identity, enable_pdl=v))
        for v in (True, False)
    )
    # Autotune record: one key per mode, otherwise identical.
    on_key, off_key = on.get_cache_key_extras(None), off.get_cache_key_extras(None)
    assert on_key[-1] == ("enable_pdl", True)
    assert off_key[-1] == ("enable_pdl", False)
    assert on_key[:-1] == off_key[:-1]
    # PDL is not a tactic axis.
    tactics = on.get_valid_tactics(None, None)
    assert tactics == off.get_valid_tactics(None, None)
    for tactic in tactics:
        on_config, off_config = (
            config_from_tactic(**identity, enable_pdl=v, tactic=tactic)
            for v in (True, False)
        )
        assert on_config != off_config
        assert replace(on_config, enable_pdl=False) == off_config
        on_name, off_name = (
            api._kernel_name(config, 5, 74, False) for config in (on_config, off_config)
        )
        assert on_name != off_name
        assert on_name.replace("_pdl1_", "_pdl0_") == off_name


def test_each_mode_loads_its_own_kernel_module():
    modules = {
        value: api._load_kernel(_base_config(enable_pdl=value))
        for value in (True, False)
    }
    for value, module in modules.items():
        assert module.enable_pdl is value
        assert sys.modules[module.__name__] is module
    assert modules[True] is not modules[False]
    assert modules[True].__name__ != modules[False].__name__


def test_load_kernel_rejects_a_non_bool_flag():
    # 1 == True, so a cached PDL-on module must not answer for it.
    api._load_kernel(_base_config(enable_pdl=True))
    with pytest.raises(TypeError, match="enable_pdl must be a bool"):
        api._load_kernel(_base_config(enable_pdl=1))


@pytest.mark.parametrize("enable_pdl", (False, True))
def test_launch_dense_gemm_builds_the_requested_mode(monkeypatch, enable_pdl):
    # The autotuner path: the identity carries the flag into every config the
    # runner builds, for the fallback and for a tuned tactic.
    configs, runners = [], []
    monkeypatch.setattr(api, "_launch", lambda config, *_, **__: configs.append(config))
    tactic = []

    def choose_one(custom_op, candidates, tuning_config, inputs, **kwargs):
        runners.extend(candidates)
        return candidates[0], tactic[0] if tactic else -1

    monkeypatch.setattr(AutoTuner.get(), "choose_one", choose_one)
    a, sfa = _fp4_activation()
    weight, sfw = _fp4_weight()
    common = dict(
        arch=100,
        operand_format="nvfp4_e2m1",
        output_format="bf16",
        epilogue="linear",
        has_bias=False,
        head_dim=None,
        is_neox=None,
        has_qkv_scale=False,
        enable_pdl=enable_pdl,
        a=a,
        weight=weight,
        output=torch.empty((M, N), dtype=torch.bfloat16),
        sfa=sfa,
        sfb=sfw,
    )
    runner.launch_dense_gemm("prims_ts_fp4_linear", **common)
    tactic.append(runners[0].get_valid_tactics(None, None)[-1])
    runner.launch_dense_gemm("prims_ts_fp4_linear", **common)
    assert [r.identity.enable_pdl for r in runners] == [enable_pdl, enable_pdl]
    assert [c.enable_pdl for c in configs] == [enable_pdl, enable_pdl]
    assert configs[0] != configs[1]


# Every PrimsTsGemmConfig field with a value other than the default. ab_stages
# enters the kernel name as the stage count the module resolved.
_OTHER_VALUE = {
    "arch": 103,
    "operand_format": "fp8_e4m3",
    "output_format": "fp8_e4m3",
    "epilogue": "swiglu",
    "has_bias": True,
    "head_dim": 128,
    "is_neox": False,
    "tile_m": 128,
    "tile_n": 128,
    "tile_k": 512,
    "cluster_shape": (4, 4, 1),
    "scheduler": "static",
    "shape_bucket": (1024, 256, 128),
    "has_qkv_scale": True,
    "nvfp4_mma_k": 96,
    "epilogue_warps": 8,
    "tmem_overlap": True,
    "use_tma_store": True,
    "separate_qkv_output": True,
    "enable_pdl": False,
}


def test_kernel_name_covers_every_config_field():
    # The on-disk cache keys a compiled kernel by this name and the source
    # hash only. A field missing here would let two programs share one entry.
    assert set(_OTHER_VALUE) | {"ab_stages"} == {
        f.name for f in fields(PrimsTsGemmConfig)
    }
    base = _base_config()
    name = api._kernel_name(base, 5, 74, False)
    for field_name, value in _OTHER_VALUE.items():
        assert getattr(base, field_name) != value, field_name
        changed = api._kernel_name(replace(base, **{field_name: value}), 5, 74, False)
        assert changed != name, field_name
    assert api._kernel_name(base, 4, 74, False) != name
    assert api._kernel_name(base, 5, 37, False) != name
    assert api._kernel_name(base, 5, 74, True) != name


_PDL_METHODS = ("wait_griddep", "launch_griddep")


def _is_enable_pdl_guard(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Attribute)
        and test.func.attr == "const_expr"
        and len(test.args) == 1
        and isinstance(test.args[0], ast.Name)
        and test.args[0].id == "enable_pdl"
    )


def _pdl_calls(tree: ast.Module) -> list[tuple[str, str, bool]]:
    """(top-level function, method, inside an enable_pdl guard) per PDL call."""
    calls = []

    def visit(node, top, guarded):
        if isinstance(node, ast.FunctionDef) and top is None:
            top = node.name
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _PDL_METHODS
        ):
            calls.append((top, node.func.attr, guarded))
        if isinstance(node, ast.If) and _is_enable_pdl_guard(node.test):
            for child in node.body:
                visit(child, top, True)
            for child in node.orelse:
                visit(child, top, guarded)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, top, guarded)

    visit(tree, None, False)
    return calls


def test_kernel_gates_every_pdl_element_on_enable_pdl():
    # One compile-time flag decides the launch attribute, every wait and the
    # release; with it off the kernel emits none of them.
    source = Path(api.__file__).with_name("kernel.py").read_text()
    tree = ast.parse(source)
    defaults = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "enable_pdl" for t in node.targets)
    ]
    assert len(defaults) == 1
    assert isinstance(defaults[0], ast.Constant) and defaults[0].value is True

    use_pdl = [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "use_pdl"
    ]
    assert len(use_pdl) == 2
    assert all(isinstance(v, ast.Name) and v.id == "enable_pdl" for v in use_pdl)

    calls = _pdl_calls(tree)
    assert all(guarded for _, _, guarded in calls)
    waits = {top for top, method, _ in calls if method == "wait_griddep"}
    releases = {top for top, method, _ in calls if method == "launch_griddep"}
    # LoadB reads only the weights and does not wait (see create_load_b_task).
    assert waits == {"create_load_a_task", "create_store_task"}
    assert releases == {"create_load_a_task"}

    barriers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("PdlWaitBarrier", "PdlLaunchBarrier")
    ]
    guarded = [
        node.body
        for node in ast.walk(tree)
        if isinstance(node, ast.IfExp)
        and isinstance(node.test, ast.Name)
        and node.test.id == "enable_pdl"
        and isinstance(node.orelse, ast.Constant)
        and node.orelse.value is None
    ]
    assert len(barriers) == 2
    assert all(any(b is g for g in guarded) for b in barriers)


def test_store_task_waits_first_for_every_epilogue():
    # The epilogue reads positions, scales and the Q/K norm weights before its
    # first accumulator wait, so under PDL the store task waits before anything
    # else whatever the epilogue is, and the resource graph says so.
    tree = ast.parse(Path(api.__file__).with_name("kernel.py").read_text())
    (create_store_task,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_store_task"
    ]
    (schedule,) = [
        node
        for node in create_store_task.body
        if isinstance(node, ast.FunctionDef) and node.name == "store_schedule"
    ]
    first = schedule.body[0]
    assert isinstance(first, ast.If) and _is_enable_pdl_guard(first.test)
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "wait_griddep"
        for node in ast.walk(first)
    )

    (call,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_store_task"
    ]
    assert isinstance(call.args[-1], ast.Name) and call.args[-1].id == "pdl_wait"

    (upstream,) = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "pdl_upstream" for t in node.targets
        )
    ]
    assert isinstance(upstream, ast.IfExp)
    assert isinstance(upstream.test, ast.Name) and upstream.test.id == "enable_pdl"
    gmem_d_deps = [
        value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Name) and key.id == "gmem_d_resource"
    ]
    assert gmem_d_deps and all(
        any(
            isinstance(e, ast.Starred)
            and isinstance(e.value, ast.Name)
            and e.value.id == "pdl_upstream"
            for e in deps.elts
        )
        for deps in gmem_d_deps
    )
