"""Execution resources for the split SM120 MegaMoE kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


def _check_cuda(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    error = result[0]
    if int(error) != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {error!r}")
    return result[1:]


def query_sm_resource_info(
    device: Optional[int] = None,
) -> tuple[int, int, int]:
    """Return SM count, minimum partition, and co-schedule alignment."""

    from cuda.core import Device

    torch.cuda.init()
    if device is None:
        device = torch.cuda.current_device()
    sm_resource = Device(device).resources.sm
    return (
        int(sm_resource.sm_count),
        max(1, int(sm_resource.min_partition_size)),
        max(1, int(sm_resource.coscheduled_alignment)),
    )


def query_green_context_sm_counts(
    *,
    k1_sm_count: int,
    device: Optional[int] = None,
) -> tuple[int, int]:
    """Return the actual CUDA-aligned K1/K2 partition sizes."""
    from cuda.bindings import driver as cuda

    torch.cuda.init()
    if device is None:
        device = torch.cuda.current_device()
    _check_cuda(cuda.cuInit(0), "cuInit")
    (cuda_device,) = _check_cuda(cuda.cuDeviceGet(device), "cuDeviceGet")
    (sm_resource,) = _check_cuda(
        cuda.cuDeviceGetDevResource(
            cuda_device,
            cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
        ),
        "cuDeviceGetDevResource",
    )
    total_sms = int(sm_resource.sm.smCount)
    min_partition = max(1, int(sm_resource.sm.minSmPartitionSize))
    if not min_partition <= k1_sm_count <= total_sms - min_partition:
        raise ValueError(
            f"invalid K1 SM count {k1_sm_count} for {total_sms} SMs"
        )
    groups, num_groups, remainder = _check_cuda(
        cuda.cuDevSmResourceSplitByCount(1, sm_resource, 0, k1_sm_count),
        "cuDevSmResourceSplitByCount",
    )
    if int(num_groups) != 1 or len(groups) != 1:
        raise RuntimeError("CUDA did not produce one K1 SM partition.")
    counts = (int(groups[0].sm.smCount), int(remainder.sm.smCount))
    if min(counts) <= 0:
        raise RuntimeError(f"CUDA produced an empty SM partition: {counts}.")
    return counts


@dataclass
class NativeGreenContextResources:
    """Four disjoint Green Contexts for K1, TX, RX, and K2.

    CUDA does not support recursively splitting an already partitioned SM
    resource.  The general resource API is therefore used once to create all
    four groups.  TX and RX use separate communication partitions: a sender
    waiting for an ACK can therefore never prevent the matching receiver CTA
    from becoming resident (and vice versa).
    """

    k1_sm_count: int
    k2_sm_count: int
    comm_sm_count: int
    comm_tx_sm_count: int
    comm_rx_sm_count: int
    comm_streams: tuple[torch.cuda.ExternalStream, ...]
    green_contexts: tuple[Any, ...]
    execution_contexts: tuple[Any, ...]
    _raw_comm_streams: tuple[Any, ...]
    _core_resources: tuple[Any, ...]
    _driver_resources: tuple[Any, ...]
    _closed: bool = False

    @classmethod
    def create(
        cls,
        *,
        k1_sm_count: int,
        k2_sm_count: int,
        comm_tx_sm_count: int,
        comm_rx_sm_count: int,
        num_comm_streams: int = 4,
        device: Optional[int] = None,
    ) -> "NativeGreenContextResources":
        from cuda.bindings import driver as cuda
        from cuda.core import Device, SMResourceOptions

        torch.cuda.init()
        if device is None:
            device = torch.cuda.current_device()
        if num_comm_streams <= 0:
            raise ValueError("num_comm_streams must be positive")

        _check_cuda(cuda.cuInit(0), "cuInit")
        (cuda_device,) = _check_cuda(
            cuda.cuDeviceGet(device), "cuDeviceGet"
        )
        sm_resource = Device(device).resources.sm
        total_sms = int(sm_resource.sm_count)
        min_partition = max(1, int(sm_resource.min_partition_size))
        alignment = max(1, int(sm_resource.coscheduled_alignment))
        requested_counts = (
            k1_sm_count,
            comm_tx_sm_count,
            comm_rx_sm_count,
            k2_sm_count,
        )
        if sum(requested_counts) != total_sms:
            raise ValueError(
                f"K1/TX/RX/K2 counts {requested_counts} must sum to "
                f"the device SM count {total_sms}."
            )
        if min(requested_counts) < min_partition:
            raise ValueError(
                f"each Green Context needs at least {min_partition} SMs; "
                f"got {requested_counts}."
            )
        if any(count % alignment for count in requested_counts[:-1]):
            raise ValueError(
                f"K1/TX/RX counts must be multiples of {alignment}; "
                f"got {requested_counts[:-1]}."
            )

        # The driver accepts the heterogeneous split in K1, TX, RX, K2 order.
        # A catch-all K2 group avoids imposing an invalid 38-SM co-schedule
        # constraint on Blackwell while still consuming the full remainder.
        options = SMResourceOptions(
            count=[k1_sm_count, comm_tx_sm_count, comm_rx_sm_count, None],
            coscheduled_sm_count=[alignment, alignment, alignment, 2],
            backfill=True,
        )
        groups, remainder = sm_resource.split(options)
        if int(remainder.sm_count) != 0 or len(groups) != 4:
            raise RuntimeError(
                "CUDA did not produce the requested K1/TX/RX/K2 split."
            )
        actual_counts = tuple(int(group.sm_count) for group in groups)
        if min(actual_counts) <= 0:
            raise RuntimeError(
                f"CUDA produced an empty SM partition: {actual_counts}."
            )
        if actual_counts != requested_counts:
            raise RuntimeError(
                f"CUDA produced {actual_counts}, requested {requested_counts}."
            )

        driver_resources = tuple(
            cuda.CUdevResource(group.handle) for group in groups
        )
        contexts = []
        execution_contexts = []
        raw_streams = []
        try:
            for label, resource in zip(
                ("K1", "COMM_TX", "COMM_RX", "K2"), driver_resources
            ):
                (descriptor,) = _check_cuda(
                    cuda.cuDevResourceGenerateDesc([resource], 1),
                    f"cuDevResourceGenerateDesc({label})",
                )
                (green_context,) = _check_cuda(
                    cuda.cuGreenCtxCreate(
                        descriptor,
                        cuda_device,
                        cuda.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
                    ),
                    f"cuGreenCtxCreate({label})",
                )
                contexts.append(green_context)
                (execution_context,) = _check_cuda(
                    cuda.cuCtxFromGreenCtx(green_context),
                    f"cuCtxFromGreenCtx({label})",
                )
                execution_contexts.append(execution_context)

            for index in range(num_comm_streams):
                priority = -1 if index < 2 else 0
                # Runner order is dispatch-TX, dispatch-RX, combine-TX,
                # combine-RX.  Even streams use TX; odd streams use RX.
                context_index = 1 if index % 2 == 0 else 2
                (raw_stream,) = _check_cuda(
                    cuda.cuGreenCtxStreamCreate(
                        contexts[context_index],
                        cuda.CUstream_flags.CU_STREAM_NON_BLOCKING,
                        priority,
                    ),
                    f"cuGreenCtxStreamCreate(COMM:{index})",
                )
                raw_streams.append(raw_stream)
        except Exception:
            for raw_stream in reversed(raw_streams):
                cuda.cuStreamDestroy(raw_stream)
            for context in reversed(contexts):
                cuda.cuGreenCtxDestroy(context)
            raise

        external_streams = tuple(
            torch.cuda.ExternalStream(int(raw_stream), device=device)
            for raw_stream in raw_streams
        )
        return cls(
            k1_sm_count=actual_counts[0],
            comm_sm_count=actual_counts[1] + actual_counts[2],
            comm_tx_sm_count=actual_counts[1],
            comm_rx_sm_count=actual_counts[2],
            k2_sm_count=actual_counts[3],
            comm_streams=external_streams,
            green_contexts=tuple(contexts),
            execution_contexts=tuple(execution_contexts),
            _raw_comm_streams=tuple(raw_streams),
            _core_resources=tuple(groups),
            _driver_resources=driver_resources,
        )

    def close(self) -> None:
        if self._closed:
            return
        from cuda.bindings import driver as cuda

        for stream in self.comm_streams:
            stream.synchronize()
        for index, raw_stream in reversed(
            list(enumerate(self._raw_comm_streams))
        ):
            _check_cuda(
                cuda.cuStreamDestroy(raw_stream),
                f"cuStreamDestroy(COMM:{index})",
            )
        for label, context in reversed(
            list(
                zip(
                    ("K1", "COMM_TX", "COMM_RX", "K2"),
                    self.green_contexts,
                )
            )
        ):
            _check_cuda(
                cuda.cuGreenCtxDestroy(context),
                f"cuGreenCtxDestroy({label})",
            )
        self._closed = True


@dataclass
class NativeGreenContextGraph:
    """A native CUDA Graph with K1/K2 nodes bound to disjoint contexts."""

    root_stream: torch.cuda.Stream
    k1_sm_count: int
    k2_sm_count: int
    _graph: Any
    _graph_exec: Any
    _green_contexts: tuple[Any, Any]
    _owns_green_contexts: bool
    _launch_ready: torch.cuda.Event
    _launch_done: torch.cuda.Event
    _closed: bool = False

    @staticmethod
    def _function_handles(executor: Any) -> set[int]:
        context = getattr(executor, "exec_context", None)
        functions = getattr(context, "kernel_functions", None)
        if not functions:
            # CUDA-dialect executors load cudaKernel_t through the generated
            # host shim and do not expose driver CUfunction objects here.
            return set()
        return {int(function) for function in functions}

    @classmethod
    def capture(
        cls,
        *,
        root_stream: torch.cuda.Stream,
        k1_stream: torch.cuda.Stream,
        k2_stream: torch.cuda.Stream,
        k1_executor: Any,
        k2_executor: Any,
        k2_drain_executor: Optional[Any],
        k3_executor: Optional[Any],
        launch_k1: Callable[[], None],
        launch_k2: Callable[[], None],
        launch_k2_drain: Optional[Callable[[], None]],
        launch_k2_finalizer: Optional[Callable[[], None]],
        launch_k3: Optional[Callable[[], None]],
        launch_reset: Optional[Callable[[], None]],
        k1_sm_count: int,
        k2_grid_blocks: Optional[int] = None,
        green_resources: Optional[NativeGreenContextResources] = None,
        device: Optional[int] = None,
    ) -> "NativeGreenContextGraph":
        """Capture on ordinary streams, then rebind K1/K2 graph nodes."""
        from cuda.bindings import driver as cuda
        from cuda.bindings import runtime as cudart

        torch.cuda.init()
        if device is None:
            device = torch.cuda.current_device()

        raw_root = cudart.cudaStream_t(root_stream.cuda_stream)
        capture_ready = torch.cuda.Event()
        k1_done = torch.cuda.Event()
        k2_done = torch.cuda.Event()
        torch.cuda.synchronize()

        _check_cuda(
            cudart.cudaStreamBeginCapture(
                raw_root,
                cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal,
            ),
            "cudaStreamBeginCapture",
        )
        try:
            if launch_reset is not None:
                with torch.cuda.stream(root_stream):
                    launch_reset()
            capture_ready.record(root_stream)
            k1_stream.wait_event(capture_ready)
            k2_stream.wait_event(capture_ready)
            launch_k1()
            launch_k2()
            if launch_k2_drain is not None:
                # Same G1 stream: the drain worker begins only after K1 has
                # released its SM partition.
                launch_k2_drain()
            k1_done.record(k1_stream)
            k2_done.record(k2_stream)
            root_stream.wait_event(k1_done)
            root_stream.wait_event(k2_done)
            if launch_k2_finalizer is not None:
                # The worker kernels deliberately skip the global tail.
                # This one-CTA node reuses the original rank barrier/reset
                # after both shared-queue consumers have drained.
                launch_k2_finalizer()
            if launch_k3 is not None:
                launch_k3()
            (graph,) = _check_cuda(
                cudart.cudaStreamEndCapture(raw_root),
                "cudaStreamEndCapture",
            )
        except Exception:
            # End capture to release stream capture state when the host wrapper
            # failed after begin-capture. The original error remains primary.
            cudart.cudaStreamEndCapture(raw_root)
            raise

        k1_handles = cls._function_handles(k1_executor)
        if k2_drain_executor is not None:
            k1_handles.update(cls._function_handles(k2_drain_executor))
        k2_handles = cls._function_handles(k2_executor)
        if k1_handles and k2_handles and k1_handles & k2_handles:
            _check_cuda(cuda.cuGraphDestroy(graph), "cuGraphDestroy")
            raise RuntimeError("K1 and K2 CUfunction handle sets overlap.")

        _check_cuda(cuda.cuInit(0), "cuInit")
        (cuda_device,) = _check_cuda(cuda.cuDeviceGet(device), "cuDeviceGet")
        owns_green_contexts = green_resources is None
        if green_resources is None:
            (sm_resource,) = _check_cuda(
                cuda.cuDeviceGetDevResource(
                    cuda_device,
                    cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
                ),
                "cuDeviceGetDevResource",
            )
            total_sms = int(sm_resource.sm.smCount)
            min_partition = max(1, int(sm_resource.sm.minSmPartitionSize))
            if not min_partition <= k1_sm_count <= total_sms - min_partition:
                raise ValueError(
                    f"invalid K1 SM count {k1_sm_count} for {total_sms} SMs"
                )
            groups, num_groups, remainder = _check_cuda(
                cuda.cuDevSmResourceSplitByCount(
                    1, sm_resource, 0, k1_sm_count
                ),
                "cuDevSmResourceSplitByCount",
            )
            if int(num_groups) != 1 or len(groups) != 1:
                _check_cuda(cuda.cuGraphDestroy(graph), "cuGraphDestroy")
                raise RuntimeError("CUDA did not produce one K1 SM partition.")
            resources = (groups[0], remainder)
            k1_sm_count = int(resources[0].sm.smCount)
            k2_sm_count = int(resources[1].sm.smCount)
            green_contexts = []
            execution_contexts = []
        else:
            resources = None
            k1_sm_count = green_resources.k1_sm_count
            k2_sm_count = green_resources.k2_sm_count
            green_contexts = [
                green_resources.green_contexts[0],
                green_resources.green_contexts[3],
            ]
            execution_contexts = [
                green_resources.execution_contexts[0],
                green_resources.execution_contexts[3],
            ]
        if k2_grid_blocks is None:
            k2_grid_blocks = k2_sm_count
        if k2_grid_blocks <= 0:
            raise ValueError("k2_grid_blocks must be positive")
        try:
            if green_resources is None:
                for index, resource in enumerate(resources):
                    (descriptor,) = _check_cuda(
                        cuda.cuDevResourceGenerateDesc([resource], 1),
                        f"cuDevResourceGenerateDesc(K{index + 1})",
                    )
                    (green_context,) = _check_cuda(
                        cuda.cuGreenCtxCreate(
                            descriptor,
                            cuda_device,
                            cuda.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
                        ),
                        f"cuGreenCtxCreate(K{index + 1})",
                    )
                    green_contexts.append(green_context)
                    (execution_context,) = _check_cuda(
                        cuda.cuCtxFromGreenCtx(green_context),
                        f"cuCtxFromGreenCtx(K{index + 1})",
                    )
                    execution_contexts.append(execution_context)

            nodes, num_nodes = _check_cuda(
                cuda.cuGraphGetNodes(graph, 0),
                "cuGraphGetNodes(count)",
            )
            nodes, num_nodes = _check_cuda(
                cuda.cuGraphGetNodes(graph, num_nodes),
                "cuGraphGetNodes",
            )
            rebound = [0, 0]
            for node in nodes:
                (node_type,) = _check_cuda(
                    cuda.cuGraphNodeGetType(node),
                    "cuGraphNodeGetType",
                )
                if (
                    node_type
                    != cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
                ):
                    continue
                (params,) = _check_cuda(
                    cuda.cuGraphKernelNodeGetParams(node),
                    "cuGraphKernelNodeGetParams",
                )
                function = int(params.func)
                grid_blocks = (
                    int(params.gridDimX)
                    * int(params.gridDimY)
                    * int(params.gridDimZ)
                )
                if k1_handles or k2_handles:
                    context_index = (
                        0 if function in k1_handles
                        else 1 if function in k2_handles
                        else None
                    )
                else:
                    # Public CuTeDSL 4.6 uses a CUDA-dialect host shim whose
                    # executor does not expose CUfunction handles. K1/K2 are
                    # persistent cluster=1 kernels, so their captured grid
                    # volumes are exactly their disjoint SM allocations.
                    context_index = (
                        0 if grid_blocks == k1_sm_count
                        else 1
                        if grid_blocks == k2_grid_blocks
                        else None
                    )
                if context_index is None:
                    continue

                updated = cuda.CUgraphNodeParams()
                updated.type = (
                    cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
                )
                updated.kernel.func = params.func
                updated.kernel.gridDimX = params.gridDimX
                updated.kernel.gridDimY = params.gridDimY
                updated.kernel.gridDimZ = params.gridDimZ
                updated.kernel.blockDimX = params.blockDimX
                updated.kernel.blockDimY = params.blockDimY
                updated.kernel.blockDimZ = params.blockDimZ
                updated.kernel.sharedMemBytes = params.sharedMemBytes
                updated.kernel.kernelParams = params.kernelParams
                updated.kernel.extra = params.extra
                updated.kernel.ctx = execution_contexts[context_index]
                _check_cuda(
                    cuda.cuGraphNodeSetParams(node, updated),
                    f"cuGraphNodeSetParams(K{context_index + 1})",
                )
                rebound[context_index] += 1

            if min(rebound) <= 0:
                raise RuntimeError(
                    "Failed to identify both CuTeDSL K1/K2 graph nodes: "
                    f"rebound={tuple(rebound)}."
                )

            instantiate_params = cuda.CUDA_GRAPH_INSTANTIATE_PARAMS()
            (graph_exec,) = _check_cuda(
                cuda.cuGraphInstantiateWithParams(
                    graph, instantiate_params
                ),
                "cuGraphInstantiateWithParams",
            )
        except Exception:
            if owns_green_contexts:
                for green_context in reversed(green_contexts):
                    cuda.cuGreenCtxDestroy(green_context)
            cuda.cuGraphDestroy(graph)
            raise

        return cls(
            root_stream=root_stream,
            k1_sm_count=k1_sm_count,
            k2_sm_count=k2_sm_count,
            _graph=graph,
            _graph_exec=graph_exec,
            _green_contexts=(green_contexts[0], green_contexts[1]),
            _owns_green_contexts=owns_green_contexts,
            _launch_ready=torch.cuda.Event(),
            _launch_done=torch.cuda.Event(),
        )

    def launch(self, caller_stream: torch.cuda.Stream) -> None:
        """Replay after prior caller-stream resets and rejoin that stream."""
        from cuda.bindings import driver as cuda

        caller_cuda = cuda.CUstream(caller_stream.cuda_stream)
        if torch.cuda.is_current_stream_capturing():
            (
                capture_status,
                _capture_id,
                capture_graph,
                dependencies,
                _edge_data,
                num_dependencies,
            ) = _check_cuda(
                cuda.cuStreamGetCaptureInfo(caller_cuda),
                "cuStreamGetCaptureInfo",
            )
            if (
                capture_status
                != cuda.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE
            ):
                raise RuntimeError(
                    "the current torch stream reported capture without an active "
                    "CUDA capture graph"
                )
            (child_node,) = _check_cuda(
                cuda.cuGraphAddChildGraphNode(
                    capture_graph,
                    dependencies,
                    num_dependencies,
                    self._graph,
                ),
                "cuGraphAddChildGraphNode",
            )
            _check_cuda(
                cuda.cuStreamUpdateCaptureDependencies(
                    caller_cuda,
                    [child_node],
                    None,
                    1,
                    cuda.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES,
                ),
                "cuStreamUpdateCaptureDependencies",
            )
            return

        self._launch_ready.record(caller_stream)
        self.root_stream.wait_event(self._launch_ready)
        _check_cuda(
            cuda.cuGraphLaunch(
                self._graph_exec,
                cuda.CUstream(self.root_stream.cuda_stream),
            ),
            "cuGraphLaunch",
        )
        self._launch_done.record(self.root_stream)
        caller_stream.wait_event(self._launch_done)

    def close(self) -> None:
        if self._closed:
            return

        from cuda.bindings import driver as cuda

        self.root_stream.synchronize()
        _check_cuda(
            cuda.cuGraphExecDestroy(self._graph_exec),
            "cuGraphExecDestroy",
        )
        _check_cuda(cuda.cuGraphDestroy(self._graph), "cuGraphDestroy")
        if self._owns_green_contexts:
            for index, green_context in reversed(
                list(enumerate(self._green_contexts))
            ):
                _check_cuda(
                    cuda.cuGreenCtxDestroy(green_context),
                    f"cuGreenCtxDestroy(K{index + 1})",
                )
        self._closed = True
