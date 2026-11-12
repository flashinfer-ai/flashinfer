"""
Copyright (c) 2024 by FlashInfer team.

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

import functools
import inspect
import threading
import weakref
from typing import Tuple

import torch

from ..jit.cubin_loader import setup_cubin_loader
from ..jit import gen_cudnn_fmha_module


_attention_handles = threading.local()
_get_current_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def supports_ordered_cudnn_execution(graph_type):
    """Detect the optional tensor-sequence overload once, independent of engine."""
    return _supports_ordered_execute(getattr(graph_type, "execute", None))


@functools.cache
def _supports_ordered_execute(execute):
    try:
        return "tensor_uids" in inspect.signature(execute).parameters
    except (TypeError, ValueError):
        # Older native graph classes may not expose an inspectable signature.
        return False


@functools.cache
def supports_native_cudnn_log2(backend, device):
    """Probe headers and runtime once without excluding backend candidates.

    A runtime version check alone is insufficient: FE built with older headers
    can accept the Python flag but decline every backend plan. Do not turn that
    into an accidental FROST-only routing policy.
    """
    if (
        device.type != "cuda"
        or backend.backend_version() < 92700
        or not hasattr(backend.pygraph, "backend_plan_entries")
    ):
        return False
    stream = torch.cuda.current_stream(device)
    graph = backend.pygraph(
        handle=get_cudnn_attention_handle(backend, stream),
        io_data_type=backend.data_type.HALF,
        intermediate_data_type=backend.data_type.FLOAT,
        compute_data_type=backend.data_type.FLOAT,
    )
    tensors = [
        graph.tensor(dim=[1, 1, 16, 64], stride=[1024, 1024, 64, 1]) for _ in range(3)
    ]
    try:
        out, stats = graph.sdpa(
            q=tensors[0],
            k=tensors[1],
            v=tensors[2],
            generate_stats=True,
            stats_use_log2=True,
        )
        out.set_output(True)
        stats.set_output(True).set_data_type(backend.data_type.FLOAT)
        graph.validate()
        graph.create_execution_plans([backend.heur_mode.A])
        return bool(graph.backend_plan_entries())
    except backend.cudnnGraphNotSupportedError:
        return False
    except TypeError as exc:
        message = str(exc)
        if "stats_use_log2" not in message or not any(
            reason in message
            for reason in ("unexpected", "incompatible function arguments")
        ):
            raise
        return False


def build_cudnn_graph_with_log2(backend, build, args, kwargs, stats_use_log2):
    """Fall back only on a graph capability decline, before any execution."""
    if stats_use_log2:
        try:
            return build(*args, stats_use_log2=True, **kwargs)
        except backend.cudnnGraphNotSupportedError:
            pass
    return build(*args, **kwargs)


def require_native_cudnn_log2(graph, backend):
    # Check the real graph too: a small capability probe does not establish
    # support for every paged/mask/layout combination. Keep the old route when
    # the backend declines, even if a FROST plan could serve log2 by itself.
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([backend.heur_mode.A])
    if not graph.backend_plan_entries():
        raise backend.cudnnGraphNotSupportedError(
            "native log2 Stats would exclude the cuDNN backend"
        )
    graph.check_support()
    graph.build_plans()
    graph._flashinfer_stats_use_log2 = True


class _AttentionHandle:
    def __init__(self, backend):
        self.value = backend.create_handle()
        weakref.finalize(self, backend.destroy_handle, self.value)


def get_cudnn_attention_handle(backend, stream=None, *, device=None):
    """A handle belongs to its CUDA device and must not race across threads."""
    if stream is not None:
        if device is not None:
            raise ValueError("pass either stream or device, not both")
        device, stream_ptr = stream.device, stream.cuda_stream
    elif (
        _get_current_raw_stream is not None
        and isinstance(device, torch.device)
        and device.type == "cuda"
        and device.index is not None
        and torch.cuda.is_initialized()
    ):
        # The caller already has a CUDA tensor on this explicit device. Read
        # its current stream each time without constructing a Python Stream.
        # Keep lazy initialization/bad-fork checks and older Torch support on
        # the ordinary path below; a raw query does not perform lazy init.
        stream_ptr = _get_current_raw_stream(device.index)
    else:
        # Warm callers already know the index; preserve the ordinary path's
        # integer lookup when the raw getter is unavailable.
        if isinstance(device, torch.device) and device.type == "cuda":
            device = device.index
        stream = torch.cuda.current_stream(device)
        device, stream_ptr = stream.device, stream.cuda_stream
    handles = getattr(_attention_handles, "handles", None)
    if handles is None:
        handles = _attention_handles.handles = {}
    owner = handles.get(device)
    if owner is None:
        with torch.cuda.device(device):
            owner = handles[device] = _AttentionHandle(backend)
    backend.set_stream(owner.value, stream_ptr)
    return owner.value


@functools.cache
def get_cudnn_fmha_gen_module():
    mod = gen_cudnn_fmha_module()
    op = mod.build_and_load()
    for library_path in mod.get_library_paths():
        setup_cubin_loader(library_path)
    return op


def cudnn_frontend_serves_frost_decode(compute_capability: Tuple[int, int]) -> bool:
    """Whether an importable supported FE serves this device's decode tile.

    FE >=1.32 includes the SM100/SM103 default FROST decode paths. Runtime
    importability and device support remain independent of that package floor.
    """
    try:
        import cudnn  # noqa: PLC0415 -- keep runtime availability checks lazy
    except Exception:  # noqa: BLE001 -- an import failure means unavailable
        return False
    return cudnn is not None and tuple(compute_capability) in ((10, 0), (10, 3))


@functools.cache
def cudnn_frontend_frost_runtime_available() -> bool:
    """Whether the installed cudnn-frontend can run its FROST (CuTe-DSL) SDPA
    engines on this install: the CuTe DSL they need is present and not below
    the frontend's own floor (``cudnn.frost.buffers.CUTEDSL_MIN_VERSION``,
    4.7.0 today).

    Consult the frontend's own runtime check rather than inferring capability
    from package metadata. Its kernel floor can rise independently, and a
    missing or broken DSL import must still leave automatic selection on its
    fallback. Internal DSL builds follow the frontend's policy as well.
    """
    try:
        from cudnn.frost.buffers import (  # noqa: PLC0415 -- optional dependency
            cutedsl_state,
            cutedsl_too_old,
        )
    except Exception:  # noqa: BLE001 -- any import failure means "not available"
        return False
    try:
        installed, version = cutedsl_state()
        return bool(installed) and not cutedsl_too_old(version)
    except Exception:  # noqa: BLE001 -- a probe that fails is a runtime that is not there
        return False
