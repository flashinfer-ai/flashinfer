"""SM120 Green Context resources and CUDA role graphs.

The role kernels are CuTe DSL cubins.  A CUDA library kernel plus an explicit
context is required for Graph kernel nodes: ``func`` remains null.
"""

from __future__ import annotations

import atexit
import ctypes
from dataclasses import dataclass
from functools import cached_property, lru_cache, wraps
import hashlib
import os
import re
import threading
from typing import Sequence

import torch
from cuda.bindings import driver as cu


def _ok(result):
    if result[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def _get_env_int(name: str, default: int) -> int:
    """Parse a signed decimal int32 environment value without whitespace."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if re.fullmatch(r"-?[0-9]+", raw) is None:
        raise ValueError(f"invalid value for {name}: {raw}")
    value = int(raw)
    if not -(1 << 31) <= value < (1 << 31):
        raise ValueError(f"invalid value for {name}: {raw}")
    return value


class _GreenResources:
    _UNSUPPORTED_RESOURCE = (cu.CUresult.CUDA_ERROR_INVALID_RESOURCE_TYPE,
                             cu.CUresult.CUDA_ERROR_NOT_SUPPORTED)

    def __init__(self):
        required = (
            'cuDeviceGetDevResource', 'cuDevSmResourceSplitByCount',
            'cuCtxGetDevResource', 'cuDevResourceGenerateDesc',
            'cuGreenCtxCreate', 'cuGreenCtxGetDevResource', 'CUdevResource',
            'CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM',
            'CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG',
            'CUdevWorkqueueConfigScope.CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED',
            'CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM',
        )
        for name in required:
            value = cu
            for component in name.split('.'):
                if not hasattr(value, component):
                    raise RuntimeError(f'cuda.bindings.driver is missing {name}')
                value = getattr(value, component)
        resource = cu.CUdevResource()
        for name in ('sm.smCount', 'sm.minSmPartitionSize',
                     'sm.smCoscheduledAlignment', 'wqConfig.sharingScope',
                     'wqConfig.wqConcurrencyLimit', 'nextResource'):
            value = resource
            for component in name.split('.'):
                if not hasattr(value, component):
                    raise RuntimeError(f'cuda.bindings.CUdevResource is missing {name}')
                value = getattr(value, component)
        self._SM = cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        self._WORKQUEUE_CONFIG = cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG
        self._GREEN_CTX_BALANCED = cu.CUdevWorkqueueConfigScope.CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED

    def get_sm_split(self, device, requested_dispatch_sms):
        whole = _ok(cu.cuDeviceGetDevResource(device, self._SM))
        if whole.type != self._SM:
            raise RuntimeError('CUDA returned a non-SM device resource')
        total = whole.sm.smCount
        alignment = max(2, whole.sm.smCoscheduledAlignment)
        minimum = max(2, whole.sm.minSmPartitionSize)
        if not (2 <= requested_dispatch_sms < total):
            raise ValueError('invalid dispatch SM request')
        compute_target = ((total - requested_dispatch_sms) // alignment) * alignment
        if compute_target < minimum:
            raise ValueError('compute partition is smaller than the CUDA split minimum')
        compute, groups, communication = _ok(
            cu.cuDevSmResourceSplitByCount(1, whole, 0, compute_target))
        if (groups != 1 or len(compute) != 1 or
                compute[0].type != self._SM or communication.type != self._SM):
            raise RuntimeError('CUDA returned an invalid Green SM split')
        return total, compute[0], communication

    def get_balanced_workqueue(self):
        primary = _ok(cu.cuCtxGetCurrent())
        if not int(primary):
            raise RuntimeError('Green workqueue query needs a primary CUDA context')
        result = cu.cuCtxGetDevResource(primary, self._WORKQUEUE_CONFIG)
        if result[0] in self._UNSUPPORTED_RESOURCE:
            return None
        resource = _ok(result)
        if resource.type != self._WORKQUEUE_CONFIG or resource.wqConfig.wqConcurrencyLimit < 2:
            raise RuntimeError('CUDA workqueue resource cannot support two Green workers')
        resource.wqConfig.sharingScope = self._GREEN_CTX_BALANCED
        resource.wqConfig.wqConcurrencyLimit = 1
        return resource

    def make_green(self, device, sm_resource, workqueue):
        resources = [sm_resource] if workqueue is None else [sm_resource, workqueue]
        descriptor = _ok(cu.cuDevResourceGenerateDesc(resources, len(resources)))
        green = _ok(cu.cuGreenCtxCreate(
            descriptor, device, cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
        if not int(green):
            raise RuntimeError('CUDA returned a null Green context')
        # GreenContextPair owns the one matching cuGreenCtxDestroy.
        return green

    def verify_balanced_workqueue(self, green):
        result = cu.cuGreenCtxGetDevResource(green, self._WORKQUEUE_CONFIG)
        if result[0] in self._UNSUPPORTED_RESOURCE:
            return
        actual = _ok(result)
        if (actual.type != self._WORKQUEUE_CONFIG or
                actual.wqConfig.sharingScope != self._GREEN_CTX_BALANCED or
                actual.wqConfig.wqConcurrencyLimit != 1):
            raise RuntimeError('Green context did not retain balanced workqueue limits')


@lru_cache(maxsize=1)
def _green_resources():
    return _GreenResources()


@dataclass(frozen=True)
class RoleBinary:
    role: int
    cubin: bytes
    smem_bytes: int
    threads: int
    grid: int


class _KernelArgs:
    def __init__(self, values: Sequence[ctypes._SimpleCData]):
        self.values = tuple(values)
        pointers = [ctypes.addressof(value) for value in self.values]
        self.pointers = (ctypes.c_void_p * len(pointers))(*pointers)

    @property
    def pointer(self) -> int:
        return ctypes.addressof(self.pointers)

    @property
    def signature(self) -> bytes:
        return b"".join(ctypes.string_at(ctypes.addressof(x), ctypes.sizeof(x)) for x in self.values)


def _child_key(binaries: Sequence[RoleBinary], arguments: Sequence[_KernelArgs]) -> bytes:
    key = hashlib.sha256()
    for binary, role_args in zip(binaries, arguments):
        key.update(len(role_args.signature).to_bytes(4, 'little'))
        key.update(role_args.signature)
        key.update(hashlib.sha256(binary.cubin).digest())
        key.update(bytes((binary.role,)))
        key.update(binary.smem_bytes.to_bytes(8, 'little'))
        key.update(binary.threads.to_bytes(4, 'little'))
        key.update(binary.grid.to_bytes(4, 'little'))
    return key.digest()


@dataclass
class _LoadedKernel:
    library: object
    kernel: object = None

    def close(self):
        if self.library is not None:
            _ok(cu.cuLibraryUnload(self.library))
            self.library = None
            self.kernel = None


_LIBRARY_LOCK = threading.RLock()
_LOADED_KERNELS: dict[tuple, _LoadedKernel] = {}
_LIVE_PAIRS: dict[int, 'GreenContextPair'] = {}
_RUNTIME_STATE = 'open'


def _runtime_locked(function):
    @wraps(function)
    def locked(*args, **kwargs):
        with _LIBRARY_LOCK:
            return function(*args, **kwargs)
    return locked


class _ChildGraph:
    def __init__(self, owner: 'GreenContextPair', binaries: Sequence[RoleBinary],
                 arguments: Sequence[_KernelArgs]):
        roles = tuple(binary.role for binary in binaries)
        if roles != (1, 2, 3):
            raise ValueError('expected SM120 dispatch, compute and finalize roles')
        if len(arguments) != len(binaries):
            raise ValueError('expected one argument vector per role')
        self.graph = None
        self.executable = None
        self.kernels = []
        self.arguments = tuple(arguments)
        try:
            self.graph = _ok(cu.cuGraphCreate(0))
            nodes = {}
            for index, binary in enumerate(binaries):
                loaded = owner._get_loaded_kernel(binary)
                self.kernels.append(loaded)
                kernel = loaded.kernel
                _ok(cu.cuKernelSetAttribute(
                    cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    binary.smem_bytes, kernel, owner.device))
                owner._check_kernel(binary, kernel)
                params = cu.CUgraphNodeParams()
                params.type = cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
                params.kernel.kern = kernel
                params.kernel.ctx = (owner.primary_context
                                     if owner.primary_ctx_roles else
                                     owner.contexts[0 if index == 0 else 1])
                params.kernel.gridDimX = binary.grid
                params.kernel.gridDimY = params.kernel.gridDimZ = 1
                params.kernel.blockDimX = binary.threads
                params.kernel.blockDimY = params.kernel.blockDimZ = 1
                params.kernel.sharedMemBytes = binary.smem_bytes
                params.kernel.kernelParams = arguments[index].pointer
                edge_data = None
                if binary.role == 3:
                    dependencies = list(nodes.values())
                elif owner.primary_ctx_roles and binary.role == 2:
                    # Give every dispatch CTA a seat before compute starts.
                    # Routed still acquires each expert's data from dispatch.
                    dependencies = [nodes[1]]
                    edge = cu.CUgraphEdgeData()
                    edge.from_port = cu.CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC
                    edge.type = cu.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC
                    edge_data = [edge]
                else:
                    dependencies = []
                nodes[binary.role] = _ok(cu.cuGraphAddNode(
                    self.graph, dependencies, edge_data, len(dependencies), params))
        except Exception:
            try:
                self.close()
            except Exception:
                owner._failed_children.append(self)
                owner._state = 'failed'
                _pin_failed_pair(owner)
                raise
            raise

    def instantiate(self):
        if self.executable is None:
            self.executable = _ok(cu.cuGraphInstantiate(self.graph, 0))

    def close(self):
        if self.executable is not None:
            _ok(cu.cuGraphExecDestroy(self.executable))
            self.executable = None
        if self.graph is not None:
            _ok(cu.cuGraphDestroy(self.graph))
            self.graph = None
        self.kernels.clear()


class GreenContextPair:
    """SM120 communication/compute split and graph ownership."""

    @_runtime_locked
    def __init__(self, device_index=None, requested_dispatch_sms=None):
        if _RUNTIME_STATE != 'open':
            raise RuntimeError(f'kernel runtime is {_RUNTIME_STATE}')
        torch.cuda.init()
        self.primary_context = _ok(cu.cuCtxGetCurrent())
        self.primary_ctx_roles = _get_env_int('GC_MOE_PRIMARY_CTX', 0) != 0
        if requested_dispatch_sms is None:
            requested_dispatch_sms = _get_env_int('GC_MOE_SM120_DISPATCH_SMS', 6)
        if device_index is None:
            device_index = torch.cuda.current_device()
        self.device = _ok(cu.cuDeviceGet(device_index))
        resources = _green_resources()
        total, compute, dispatch = resources.get_sm_split(self.device, requested_dispatch_sms)
        self.compute_sms = compute.sm.smCount
        self.dispatch_sms = dispatch.sm.smCount
        if (self.compute_sms < 32 or self.compute_sms % 2 or
                self.dispatch_sms <= 1 or self.compute_sms + self.dispatch_sms != total):
            raise RuntimeError('unsupported Green SM partition')
        parts = () if self.primary_ctx_roles else (dispatch, compute)
        workqueue = (resources.get_balanced_workqueue()
                     if parts and _get_env_int('GC_MOE_BALANCED_WORKQUEUES', 1) else None)
        self.balanced_workqueues = workqueue is not None
        self.green = []
        self.contexts = []
        self.streams = []
        self.retirement = []
        self.graphs = {}
        self._failed_libraries = []
        self._failed_children = []
        self._state = 'open'
        _LIVE_PAIRS[id(self)] = self
        try:
            for part in parts:
                green = resources.make_green(self.device, part, workqueue)
                self.green.append(green)
                if workqueue is not None:
                    resources.verify_balanced_workqueue(green)
                self.contexts.append(_ok(cu.cuCtxFromGreenCtx(green)))
                self.streams.append(_ok(cu.cuGreenCtxStreamCreate(
                    green, cu.CUstream_flags.CU_STREAM_NON_BLOCKING, 0)))
                actual_green = _ok(cu.cuStreamGetGreenCtx(self.streams[-1]))
                if int(actual_green) != int(green):
                    raise RuntimeError('Green stream belongs to another context')
                self.retirement.append(_ok(cu.cuEventCreate(cu.CUevent_flags.CU_EVENT_DISABLE_TIMING)))
        except Exception:
            try:
                self.close()
            except Exception:
                _pin_failed_pair(self)
                raise
            raise

    @cached_property
    def smem_per_sm(self):
        return _ok(cu.cuDeviceGetAttribute(
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            self.device))

    @cached_property
    def reserved_smem_per_block(self):
        return _ok(cu.cuDeviceGetAttribute(
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK,
            self.device))

    def _check_kernel(self, binary: RoleBinary, kernel):
        active = _ok(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            cu.CUfunction(int(kernel)), binary.threads, binary.smem_bytes))
        sms = self.dispatch_sms if binary.role == 1 else self.compute_sms
        if binary.grid > active * sms:
            raise RuntimeError(f'SM120 role {binary.role} requires {binary.grid} resident CTAs, '
                               f'but the partition can hold {active * sms}')

    @_runtime_locked
    def _get_loaded_kernel(self, binary: RoleBinary) -> _LoadedKernel:
        if _RUNTIME_STATE != 'open':
            raise RuntimeError(f"kernel runtime is {_RUNTIME_STATE}")
        key = (binary.role, hashlib.sha256(binary.cubin).digest())
        loaded = _LOADED_KERNELS.get(key)
        if loaded is not None:
            return loaded
        library = _ok(cu.cuLibraryLoadData(binary.cubin, None, None, 0, None, None, 0))
        loaded = _LoadedKernel(library)
        try:
            if _ok(cu.cuLibraryGetKernelCount(library)) != 1:
                raise RuntimeError('expected exactly one kernel in a role library')
            kernels = _ok(cu.cuLibraryEnumerateKernels(1, library))
            if isinstance(kernels, tuple):
                kernels = kernels[0]
            if len(kernels) != 1:
                raise RuntimeError('role library enumeration changed')
            loaded.kernel = kernels[0]
            _LOADED_KERNELS[key] = loaded
            return loaded
        except Exception:
            try:
                loaded.close()
            except Exception:
                self._failed_libraries.append(loaded)
                self._state = 'failed'
                _pin_failed_pair(self)
                raise
            raise

    @_runtime_locked
    def launch(self, binaries: Sequence[RoleBinary], arguments: Sequence,
               stream: int | None = None) -> None:
        if _RUNTIME_STATE != 'open':
            raise RuntimeError(f"kernel runtime is {_RUNTIME_STATE}")
        if self._state != 'open':
            raise RuntimeError(f"Green context pair is {self._state}")
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        status, capture_status, _, parent, dependencies, edges, count = cu.cuStreamGetCaptureInfo(stream)
        if status != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cannot inspect CUDA stream capture: {status}")
        if capture_status not in (cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE,
                                  cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE):
            raise RuntimeError('invalid CUDA stream capture state')
        # A role may have its own CuTe kernel ABI (for example a peer pointer
        # table used by dispatch/epilogue but not the MMA worker). Retain the
        # previous flat-vector form for existing single-rank callers.
        if len(arguments) == len(binaries) and all(isinstance(item, (tuple, list))
                                                  for item in arguments):
            args = tuple(_KernelArgs(item) for item in arguments)
        else:
            common = _KernelArgs(arguments)
            args = (common,) * len(binaries)
        key = _child_key(binaries, args)
        graph = self.graphs.get(key)
        if graph is None:
            graph = _ChildGraph(self, binaries, args)
            try:
                self.graphs[key] = graph
            except Exception:
                try:
                    graph.close()
                except Exception:
                    self._failed_children.append(graph)
                    self._state = 'failed'
                    _pin_failed_pair(self)
                    raise
                raise
        if capture_status == cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            params = cu.CUgraphNodeParams()
            params.type = cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH
            params.graph.graph = graph.graph
            child = _ok(cu.cuGraphAddNode(parent, dependencies, edges, count, params))
            _ok(cu.cuStreamUpdateCaptureDependencies(
                stream, [child], None, 1,
                cu.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES))
        elif capture_status == cu.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
            graph.instantiate()
            _ok(cu.cuGraphLaunch(graph.executable, stream))

    @_runtime_locked
    def close(self):
        if self._state == 'closed':
            return
        if self._state != 'open':
            raise RuntimeError(f"Green context pair is {self._state}")
        self._state = 'closing'
        try:
            # Callers retire outer captured graphs first. Primary-role work
            # belongs to the caller's context, which may no longer be current
            # during thread-local teardown.
            if self.primary_ctx_roles:
                current = _ok(cu.cuCtxGetCurrent())
                switch_context = int(current) != int(self.primary_context)
                if switch_context:
                    _ok(cu.cuCtxPushCurrent(self.primary_context))
                try:
                    _ok(cu.cuCtxSynchronize())
                finally:
                    if switch_context:
                        _ok(cu.cuCtxPopCurrent())
            else:
                for green, event in zip(self.green, self.retirement):
                    status = cu.cuGreenCtxRecordEvent(green, event)[0]
                    if status == cu.CUresult.CUDA_SUCCESS:
                        _ok(cu.cuEventSynchronize(event))
                    else:
                        raise RuntimeError(f"cannot retire Green context work: {status}")
            for graph in self.graphs.values():
                graph.close()
            for graph in self._failed_children:
                graph.close()
            self.graphs.clear()
            self._failed_children.clear()
            while self._failed_libraries:
                self._failed_libraries[-1].close()
                self._failed_libraries.pop()
            for event in reversed(self.retirement):
                _ok(cu.cuEventDestroy(event))
            for stream in reversed(self.streams):
                _ok(cu.cuStreamDestroy(stream))
            for green in reversed(self.green):
                _ok(cu.cuGreenCtxDestroy(green))
            self.retirement.clear()
            self.streams.clear()
            self.contexts.clear()
            self.green.clear()
        except Exception:
            # A partially retired pair must never be returned from the
            # thread-local cache or used for another launch.
            self._state = 'failed'
            _pin_failed_pair(self)
            raise
        self._state = 'closed'
        _LIVE_PAIRS.pop(id(self), None)


_THREAD_LOCAL = threading.local()
_FAILED_PAIRS: dict[int, GreenContextPair] = {}


def _pin_failed_pair(pair: GreenContextPair) -> None:
    # Failed CUDA retirement has no safe retry contract. Keep live mappings
    # and child resources reachable until process teardown.
    _FAILED_PAIRS[id(pair)] = pair


@_runtime_locked
def _close_runtime():
    global _RUNTIME_STATE
    if _RUNTIME_STATE == 'closed':
        return
    if _RUNTIME_STATE != 'open':
        raise RuntimeError(f"kernel runtime is {_RUNTIME_STATE}")
    _RUNTIME_STATE = 'retiring'
    try:
        for pair in tuple(_LIVE_PAIRS.values()):
            _ok(cu.cuCtxPushCurrent(pair.primary_context))
            try:
                pair.close()
            finally:
                _ok(cu.cuCtxPopCurrent())
        while _LOADED_KERNELS:
            key = next(reversed(_LOADED_KERNELS))
            _LOADED_KERNELS[key].close()
            del _LOADED_KERNELS[key]
    except Exception:
        _RUNTIME_STATE = 'failed'
        raise
    _RUNTIME_STATE = 'closed'


atexit.register(_close_runtime)


class _ThreadContexts:
    def __init__(self):
        self.contexts: dict[int, GreenContextPair] = {}

    def __del__(self):
        # Thread-local destruction runs in the owning thread. Callers must
        # retire any outer captured graphs before this point.
        for pair in self.contexts.values():
            if pair._state == 'open':
                try:
                    pair.close()
                except Exception:
                    _pin_failed_pair(pair)
            elif pair._state == 'failed':
                _pin_failed_pair(pair)


def get_context() -> GreenContextPair:
    current = _ok(cu.cuCtxGetCurrent())
    holder = getattr(_THREAD_LOCAL, 'holder', None)
    if holder is None:
        holder = _THREAD_LOCAL.holder = _ThreadContexts()
    contexts = holder.contexts
    key = int(current)
    if key not in contexts or contexts[key]._state == 'closed':
        contexts[key] = GreenContextPair()
    elif contexts[key]._state != 'open':
        raise RuntimeError(f"cached Green context pair is {contexts[key]._state}")
    return contexts[key]


@_runtime_locked
def release_workspace_graphs(address: int):
    """Retire cached children after the caller has retired outer graphs and GPU work."""
    for pair in _LIVE_PAIRS.values():
        keys = [key for key, child in pair.graphs.items()
                if child.arguments[0].values[0].value == address]
        if keys:
            _ok(cu.cuCtxPushCurrent(pair.primary_context))
            try:
                for key in keys:
                    try:
                        pair.graphs[key].close()
                    except Exception:
                        pair._state = 'failed'
                        _pin_failed_pair(pair)
                        raise
                    del pair.graphs[key]
            finally:
                _ok(cu.cuCtxPopCurrent())
