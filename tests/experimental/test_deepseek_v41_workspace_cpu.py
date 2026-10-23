# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Guarded host contracts for V4-style DS4.1 workspace and public ownership.

Run with --confcutdir=tests/experimental to exclude root CUDA collection probes.
Real storage is always CPU-only. CUDA designation, context, capture state and
device facts are explicit host mocks; the actual Torch CUDA entry points and
accelerator imports are guarded. A surrounding runner may have initialized CUDA;
these tests must preserve that state. They do not lower or execute kernels.
"""

import ast
import builtins
from contextlib import nullcontext
from dataclasses import dataclass
import functools
import inspect
import math
import os
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_PRIVATE = _ROOT / "flashinfer/experimental/deepseek_v41"
_PUBLIC = _ROOT / "flashinfer/deepseek_v41.py"


def _load_host(path, name, monkeypatch, **bindings):
    """Execute actual host definitions, not package/CuTe import side effects."""
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    module = ModuleType(name)
    module.__dict__.update(bindings)
    module.__package__ = "flashinfer"
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


class _CudaTensor:
    """A CUDA declaration backed exclusively by a real CPU tensor."""

    def __init__(self, tensor, *, device=None, requires_grad=None):
        self.tensor = tensor
        self.device = tensor.device if device is None else device
        self.requires_grad = (
            tensor.requires_grad if requires_grad is None else requires_grad
        )

    def __getattr__(self, name):
        return getattr(self.tensor, name)


@pytest.fixture
def workspace_host(monkeypatch):
    import torch

    initialized = torch.cuda._initialized
    visibility = {
        name: (name in os.environ, os.getenv(name))
        for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
    }
    blocked, imports = [], {}

    def forbidden(label):
        def fail(*args, **kwargs):
            blocked.append(label)
            raise AssertionError("host workspace tests must not use CUDA/JIT: " + label)

        return fail

    for name in (
        "init",
        "_lazy_init",
        "is_available",
        "device_count",
        "current_device",
        "get_device_capability",
        "get_device_properties",
        "synchronize",
        "current_stream",
        "default_stream",
        "is_current_stream_capturing",
        "device",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden("torch.cuda." + name))
    for name in (
        "_cuda_init",
        "_cuda_getDeviceCount",
        "_cuda_getDevice",
        "_cuda_getDeviceProperties",
        "_cuda_isCurrentStreamCapturing",
        "_cuda_synchronize",
    ):
        if hasattr(torch._C, name):
            monkeypatch.setattr(torch._C, name, forbidden("torch._C." + name))
    # Do not import bindings merely to guard them. Unexpected future imports
    # are blocked below; already-loaded entry points are guarded directly.
    for module_name in (
        "cuda.bindings.driver",
        "cuda.bindings.runtime",
        "cuda.cuda",
        "cuda.cudart",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            for name in dir(module):
                value = getattr(module, name)
                if (
                    name.startswith("cu")
                    and callable(value)
                    and not isinstance(value, type)
                ):
                    monkeypatch.setattr(
                        module, name, forbidden(module_name + "." + name)
                    )
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            level == 1
            and globals
            and globals.get("__name__") == "_dsv41_public_workspace_cpu"
        ):
            if name in imports:
                return imports[name]
            return forbidden("unmocked public import " + name)()
        if name.split(".")[0] in {"cuda", "cutlass", "flashinfer", "triton", "nvidia"}:
            return forbidden("accelerator import " + name)()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    helpers = _load_host(
        _PRIVATE / "_utils.py", "_dsv41_workspace_utils_cpu", monkeypatch
    )
    module = _load_host(
        _PRIVATE / "prims_ts/_workspace.py",
        "_dsv41_workspace_cpu",
        monkeypatch,
        torch=torch,
        dataclass=dataclass,
        functools=functools,
        math=math,
        struct=struct,
        _overlaps=helpers._overlaps,
    )
    public = _load_host(
        _PUBLIC,
        "_dsv41_public_workspace_cpu",
        monkeypatch,
        flashinfer_experimental_api=lambda function: function,
    )
    state = SimpleNamespace(
        capturing=False,
        properties=0,
        allocations=[],
        capability=(10, 0),
        sm_count=148,
    )
    cuda_device = torch.device("cuda:0")

    def properties(index):
        state.properties += 1
        assert index == 0
        return SimpleNamespace(
            major=state.capability[0],
            minor=state.capability[1],
            multi_processor_count=state.sm_count,
        )

    def allocate(shape, *, device, dtype):
        assert device == cuda_device
        tensor = torch.empty(shape, dtype=dtype, device="cpu")
        state.allocations.append(tensor)
        return _CudaTensor(tensor, device=cuda_device)

    class HostTorch:
        cuda = SimpleNamespace(
            device=lambda device: nullcontext(),
            is_current_stream_capturing=lambda: state.capturing,
            get_device_properties=properties,
        )
        empty = staticmethod(allocate)

        def __getattr__(self, name):
            return getattr(torch, name)

    # Production global lookups see only these declared host simulations.
    # The real Torch CUDA guard remains installed and is never called.
    monkeypatch.setattr(module, "torch", HostTorch())
    imports["experimental.deepseek_v41.prims_ts._workspace"] = module
    assert torch.cuda._initialized == initialized and not blocked
    try:
        yield SimpleNamespace(
            torch=torch,
            module=module,
            public=public,
            imports=imports,
            state=state,
            cuda_device=cuda_device,
        )
    finally:
        assert torch.cuda._initialized == initialized
        assert not blocked, f"intercepted forbidden CUDA/JIT/import calls: {blocked}"
        assert visibility == {
            name: (name in os.environ, os.getenv(name)) for name in visibility
        }


def _arguments(host, *, k=64, layout="HND", batch=2):
    torch = host.torch

    def tensor(shape, dtype, fill=0x5A):
        value = torch.empty(shape, dtype=dtype, device="cpu")
        value.view(torch.uint8).fill_(fill)
        return _CudaTensor(value, device=host.cuda_device)

    cache_shape = (2, 1, 64, 512) if layout == "HND" else (2, 64, 1, 512)
    main_shape = (3, 1, 64, 288) if layout == "HND" else (3, 64, 1, 288)
    required = host.module.workspace_size(batch, k)
    return dict(
        query=tensor((batch, 1, 64, 512), torch.float8_e4m3fn),
        swa_kv_cache=tensor(cache_shape, torch.float8_e4m3fn),
        workspace_buffer=tensor((required,), torch.uint8, 0xA5),
        sparse_indices=tensor((batch, 128 + k), torch.int32),
        compressed_kv_cache=tensor(main_shape, torch.uint8) if k else None,
        sparse_topk_lens=tensor((batch,), torch.int32),
        seq_lens=tensor((batch,), torch.int32),
        out=tensor((batch, 1, 64, 512), torch.bfloat16, 0x3C),
        sinks=tensor((64,), torch.float32),
        kv_layout=layout,
        backend="prims-ts",
    )


def _snapshots(host, arguments):
    return [
        (value.tensor, value.tensor.view(host.torch.uint8).clone())
        for value in arguments.values()
        if isinstance(value, _CudaTensor)
    ]


def _assert_untouched(host, snapshots):
    for value, before in snapshots:
        assert host.torch.equal(value.view(host.torch.uint8), before)


@pytest.mark.parametrize("k", range(0, 513, 64))
def test_workspace_size_and_real_cpu_views(workspace_host, k):
    host, batch = workspace_host, 2
    size = host.module.workspace_size(batch, k)
    storage = host.torch.full((size,), 0xA5, device="cpu", dtype=host.torch.uint8)
    before = storage.clone()
    views = host.module.workspace_views(storage, batch, k)
    assert host.torch.equal(storage, before)
    sections = host.module._sections(batch, k)
    end = 0
    for name, shape, dtype in sections:
        view = getattr(views, name)
        assert tuple(view.shape) == shape and view.dtype == dtype
        assert view.data_ptr() % 32 == 0
        start = view.data_ptr() - storage.data_ptr()
        assert start >= end
        end = start + view.numel() * view.element_size()
        assert end <= size
        assert view.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    assert end == size
    assert views.lse.shape == (batch, 64, 1)
    assert [name for name, _, _ in sections] == ["lse"] + (
        ["dummy_main"] if k == 0 else []
    )
    assert set(views.__dataclass_fields__) == {"lse", "dummy_main"}
    assert size == batch * 64 * 4 + (64 * 288 if k == 0 else 0)
    assert (views.dummy_main is None) == (k > 0)
    assert host.public.deepseek_v41_decode_workspace_size(batch, k) == size
    assert host.state.properties == 0 and not host.state.allocations


@pytest.mark.parametrize("batch", (1, 4, 5, 16, 17, 32, 33, 64, 65, 65535))
def test_workspace_pure_batch_boundaries(workspace_host, batch):
    host = workspace_host
    assert type(host.module.workspace_size(batch, 512)) is int
    assert host.state.properties == 0 and not host.state.allocations


def test_signed_byte_workspace_with_nonzero_storage_offset(workspace_host):
    host = workspace_host
    size = host.module.workspace_size(2, 64, backend="prims-ts")
    owner = host.torch.full((size + 64,), 0xA5, dtype=host.torch.uint8, device="cpu")
    storage = owner[32 : 32 + size].view(host.torch.int8)
    before = owner.clone()
    views = host.module.workspace_views(storage, 2, 64, backend="prims-ts")
    assert storage.storage_offset() == 32
    for name, _, _ in host.module._sections(2, 64):
        view = getattr(views, name)
        assert view.data_ptr() % 32 == 0
        assert storage.data_ptr() <= view.data_ptr()
        assert (
            view.data_ptr() + view.numel() * view.element_size()
            <= storage.data_ptr() + size
        )
    assert host.torch.equal(owner, before)


@pytest.mark.parametrize("batch", (0, -1, 65536, 1.5, True))
def test_workspace_size_rejects_batch(workspace_host, batch):
    with pytest.raises(ValueError, match="batch_size"):
        workspace_host.module.workspace_size(batch, 0)


@pytest.mark.parametrize("k", (-64, 1, 63, 576, 64.0, True))
def test_workspace_size_rejects_topk(workspace_host, k):
    with pytest.raises(ValueError, match="compressed_topk"):
        workspace_host.module.workspace_size(1, k)


@pytest.mark.parametrize("backend", ("cute-dsl", "auto", "sparse", None))
def test_workspace_size_rejects_backend(workspace_host, backend):
    with pytest.raises(ValueError, match="backend must"):
        workspace_host.public.deepseek_v41_decode_workspace_size(1, backend=backend)


@pytest.mark.parametrize("fault", ("short", "unaligned", "strided", "dtype"))
def test_workspace_views_reject_without_writing(workspace_host, fault):
    host = workspace_host
    size = host.module.workspace_size(2, 64, backend="prims-ts")
    owner = host.torch.full(
        (size * 2 + 32,), 0xA5, dtype=host.torch.uint8, device="cpu"
    )
    value = owner[:size]
    if fault == "short":
        value = value[:-1]
    elif fault == "unaligned":
        value = owner[16 : size + 16]
    elif fault == "strided":
        value = owner[: size * 2 : 2]
    else:
        value = owner[:size].view(host.torch.float32)
    before = owner.clone()
    with pytest.raises(ValueError):
        host.module.workspace_views(value, 2, 64, backend="prims-ts")
    assert host.torch.equal(owner, before)


@pytest.mark.parametrize(
    "value", (float("nan"), float("inf"), -float("inf"), 1e100, 2**1024, True, "1")
)
@pytest.mark.parametrize("positive", (False, True))
def test_scalar_range_rejects_without_cuda(workspace_host, value, positive):
    with pytest.raises(ValueError, match="scalar|FP32"):
        workspace_host.module._scalar(value, "scale", positive=positive)


@pytest.mark.parametrize("value", (0.0, -1.0, 1e-100))
def test_bmm1_requires_positive_after_fp32_rounding(workspace_host, value):
    with pytest.raises(ValueError, match="positive finite"):
        workspace_host.module._scalar(value, "bmm1_scale", positive=True)


def test_scalar_no_tensor_and_fp32_values(workspace_host):
    host = workspace_host
    with pytest.raises(ValueError, match="Python scalar"):
        host.module._scalar(host.torch.tensor(1.0, device="cpu"), "scale")
    assert host.module._scalar(0, "bmm2") == 0.0
    assert host.module._scalar(-2.0, "bmm2") == -2.0
    value = host.module._scalar(1.0000001, "bmm1", positive=True)
    assert value == struct.unpack("f", struct.pack("f", 1.0000001))[0]


@pytest.mark.parametrize("capability", ((10, 0), (10, 3), (10, 7)))
@pytest.mark.parametrize("layout", ("HND", "NHD"))
@pytest.mark.parametrize("k", (0, 64))
def test_bind_normalizes_views_and_preserves_output_identity(
    workspace_host, capability, layout, k
):
    host = workspace_host
    host.state.capability = capability
    arguments = _arguments(host, k=k, layout=layout)
    snapshots = _snapshots(host, arguments)
    bound = host.module.bind_inputs(**arguments)
    assert bound.out is arguments["out"]
    assert bound.indices is arguments["sparse_indices"]
    assert bound.topk_lens is arguments["sparse_topk_lens"]
    assert bound.seq_lens is arguments["seq_lens"]
    assert bound.swa_cache.shape == (2, 64, 1, 512)
    assert bound.swa_cache.data_ptr() == arguments["swa_kv_cache"].data_ptr()
    if k:
        assert bound.main_cache.shape == (3, 64, 1, 288)
        assert (
            bound.main_cache.data_ptr() == arguments["compressed_kv_cache"].data_ptr()
        )
    else:
        assert bound.main_cache is bound.views.dummy_main
    assert bound.views.lse.shape == (2, 64, 1)
    assert host.state.properties == 1 and not host.state.allocations
    _assert_untouched(host, snapshots)


@pytest.mark.parametrize(
    "fault",
    (
        "indices_none",
        "topk_none",
        "seq_none",
        "indices_rank",
        "indices_width",
        "topk_dtype",
        "seq_shape",
        "main_missing",
        "out_dtype",
        "layout",
        "cache_layout",
        "workspace_short",
        "workspace_input_overlap",
        "out_input_overlap",
        "out_workspace_overlap",
        "grad",
        "device",
        "bmm1",
        "bmm2",
        "backend",
        "bf16_query",
    ),
)
def test_bind_invalid_declarations_leave_storage_untouched(workspace_host, fault):
    host = workspace_host
    arguments = _arguments(host)
    torch = host.torch
    if fault.endswith("_none"):
        key = {
            "indices_none": "sparse_indices",
            "topk_none": "sparse_topk_lens",
            "seq_none": "seq_lens",
        }[fault]
        arguments[key] = None
    elif fault == "indices_rank":
        arguments["sparse_indices"].tensor = arguments["sparse_indices"].tensor.view(
            2, 1, 192
        )
    elif fault == "indices_width":
        arguments["sparse_indices"].tensor = (
            arguments["sparse_indices"].tensor[:, :191].clone()
        )
    elif fault == "topk_dtype":
        arguments["sparse_topk_lens"].tensor = arguments[
            "sparse_topk_lens"
        ].tensor.view(torch.float32)
    elif fault == "seq_shape":
        arguments["seq_lens"].tensor = arguments["seq_lens"].tensor.view(2, 1)
    elif fault == "main_missing":
        arguments["compressed_kv_cache"] = None
    elif fault == "out_dtype":
        arguments["out"].tensor = arguments["out"].tensor.view(torch.float16)
    elif fault == "layout":
        arguments["kv_layout"] = "invalid"
    elif fault == "cache_layout":
        arguments["swa_kv_cache"].tensor = arguments["swa_kv_cache"].tensor.view(
            2, 64, 1, 512
        )
    elif fault == "workspace_short":
        arguments["workspace_buffer"].tensor = arguments["workspace_buffer"].tensor[:-1]
    elif fault == "workspace_input_overlap":
        required = host.module.workspace_size(2, 64, backend="prims-ts")
        raw = torch.empty(max(required, 2 * 192 * 4), dtype=torch.uint8, device="cpu")
        arguments["workspace_buffer"].tensor = raw[:required]
        arguments["sparse_indices"].tensor = (
            raw[: 2 * 192 * 4].view(torch.int32).view(2, 192)
        )
    elif fault == "out_input_overlap":
        # A BF16 output view of the packed main pool, with sufficient backing.
        raw = torch.empty(2 * 64 * 512 * 2, dtype=torch.uint8, device="cpu")
        arguments["out"].tensor = raw.view(torch.bfloat16).view(2, 1, 64, 512)
        arguments["compressed_kv_cache"].tensor = raw[: 3 * 64 * 288].view(
            3, 1, 64, 288
        )
    elif fault == "out_workspace_overlap":
        required = host.module.workspace_size(2, 64, backend="prims-ts")
        raw = torch.empty(
            max(required, 2 * 64 * 512 * 2), dtype=torch.uint8, device="cpu"
        )
        arguments["workspace_buffer"].tensor = raw[:required]
        arguments["out"].tensor = (
            raw[: 2 * 64 * 512 * 2].view(torch.bfloat16).view(2, 1, 64, 512)
        )
    elif fault == "grad":
        arguments["out"].requires_grad = True
    elif fault == "device":
        arguments["seq_lens"].device = torch.device("cpu")
    elif fault == "bmm1":
        arguments["bmm1_scale"] = 0.0
    elif fault == "bmm2":
        arguments["bmm2_scale"] = float("inf")
    elif fault == "backend":
        arguments["backend"] = "auto"
    elif fault == "bf16_query":
        arguments["query"].tensor = torch.empty(
            (2, 1, 64, 512), dtype=torch.bfloat16, device="cpu"
        )
    snapshots = _snapshots(host, arguments)
    with torch.enable_grad(), pytest.raises(ValueError):
        host.module.bind_inputs(**arguments)
    _assert_untouched(host, snapshots)
    assert host.state.properties == 0 and not host.state.allocations


def test_read_only_inputs_may_alias(workspace_host):
    host = workspace_host
    arguments = _arguments(host)
    arguments["swa_kv_cache"].tensor = arguments["query"].tensor
    snapshots = _snapshots(host, arguments)
    bound = host.module.bind_inputs(**arguments)
    assert bound.query.data_ptr() == bound.swa_cache.data_ptr()
    assert bound.out is arguments["out"]
    _assert_untouched(host, snapshots)


def test_bind_eager_output_and_capture_output_requirement(workspace_host):
    host = workspace_host
    arguments = _arguments(host)
    arguments["out"] = None
    host.state.capturing = True
    with pytest.raises(ValueError, match="caller-supplied out"):
        host.module.bind_inputs(**arguments)
    assert not host.state.allocations and host.state.properties == 0
    host.state.capturing = False
    bound = host.module.bind_inputs(**arguments)
    assert len(host.state.allocations) == 1
    assert bound.out.tensor is host.state.allocations[0]
    assert bound.out.dtype == host.torch.bfloat16


@pytest.mark.parametrize(
    "capability,sm_count", (((10, 0), 148), ((10, 3), 73), ((10, 7), 117))
)
def test_device_facts_cold_capture_rejected_warm_capture_cached(
    workspace_host, capability, sm_count
):
    host = workspace_host
    host.state.capability, host.state.sm_count = capability, sm_count
    assert host.module.device_sm_count.cache_parameters() == {
        "maxsize": None,
        "typed": False,
    }
    host.state.capturing = True
    with pytest.raises(ValueError, match="warm"):
        host.module.device_sm_count(0)
    assert host.state.properties == 0
    host.state.capturing = False
    assert host.module.device_sm_count(0) == sm_count
    host.state.capturing = True
    assert host.module.device_sm_count(0) == sm_count
    assert host.state.properties == 1
    assert host.module.device_sm_count.cache_info().hits == 1


@pytest.mark.parametrize("capability", ((9, 0), (10, 1), (11, 0), (12, 0)))
def test_device_facts_reject_unsupported_architecture(workspace_host, capability):
    host = workspace_host
    host.state.capability = capability
    with pytest.raises(ValueError):
        host.module.device_sm_count(0)
    assert host.state.properties == 1 and not host.state.allocations
    assert host.module.device_sm_count.cache_info().currsize == 0


@pytest.mark.parametrize("capability", ((9, 0), (10, 1), (11, 0), (12, 0)))
@pytest.mark.parametrize("caller_output", (False, True))
def test_bind_rejects_unsupported_architecture_before_allocation_or_writes(
    workspace_host, capability, caller_output
):
    host = workspace_host
    host.state.capability = capability
    arguments = _arguments(host)
    if not caller_output:
        arguments["out"] = None
    snapshots = _snapshots(host, arguments)
    with pytest.raises(ValueError):
        host.module.bind_inputs(**arguments)
    _assert_untouched(host, snapshots)
    assert host.state.properties == 1 and not host.state.allocations


def test_public_dispatcher_signature_and_backend_forwarding(workspace_host):
    host = workspace_host
    decode = host.public.deepseek_v41_decode
    parameters = inspect.signature(decode).parameters
    assert list(parameters) == ["args", "backend", "kwargs"]
    assert parameters["args"].kind == inspect.Parameter.VAR_POSITIONAL
    assert parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD
    assert parameters["backend"].default == "cute-dsl"
    assert parameters["backend"].kind == inspect.Parameter.KEYWORD_ONLY
    out, query, cache, workspace, indices, topk, seq = (object() for _ in range(7))
    legacy_result = (out, object(), object())
    for backend, suffix, entry in (
        ("cute-dsl", "decode", "decode"),
        ("prims-ts", "prims_ts", "decode_mixed"),
    ):
        expected = out if backend == "prims-ts" else legacy_result
        target = Mock(return_value=expected)
        host.imports["experimental.deepseek_v41." + suffix] = SimpleNamespace(
            **{entry: target}
        )
        if backend == "prims-ts":
            args = query, cache, workspace, indices, None, topk, seq, out
            kwargs = {"bmm1_scale": 0.25, "bmm2_scale": 2.0, "kv_layout": "NHD"}
        else:
            args = tuple(object() for _ in range(6))
            kwargs = {"plan": legacy_result[2]}
        assert decode(*args, backend=backend, **kwargs) is expected
        target.assert_called_once_with(*args, **kwargs)


def test_window_delegates_k0_output_only(workspace_host):
    host = workspace_host
    result = object()
    target = Mock(return_value=result)
    window = next(
        node
        for node in ast.parse((_PRIVATE / "prims_ts/__init__.py").read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "decode_window"
    )
    namespace = {"decode_mixed": target}
    exec(
        compile(ast.Module(body=[window], type_ignores=[]), str(_PRIVATE), "exec"),
        namespace,
    )
    host.imports["experimental.deepseek_v41.prims_ts"] = SimpleNamespace(
        decode_window=namespace["decode_window"]
    )
    args = tuple(object() for _ in range(6))
    query, cache, workspace, indices, topk, seq = args
    assert host.public.deepseek_v41_window_decode(*args, backend="prims-ts") is result
    target.assert_called_once_with(
        query,
        cache,
        workspace,
        indices,
        None,
        topk,
        seq,
        None,
        1.0,
        1.0,
        None,
        "HND",
    )
