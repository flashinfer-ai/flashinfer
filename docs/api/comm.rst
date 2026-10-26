.. _apicomm:

flashinfer.comm
===============

.. currentmodule:: flashinfer.comm

This module provides communication primitives and utilities for distributed computing, including CUDA IPC, AllReduce operations, and memory management utilities.

CUDA IPC Utilities
------------------

.. autosummary::
    :toctree: ../generated

    CudaRTLibrary
    create_shared_buffer
    free_shared_buffer

DLPack Utilities
----------------

.. autosummary::
    :toctree: ../generated

    pack_strided_memory

Mapping Utilities
-----------------

.. autosummary::
    :toctree: ../generated

    Mapping

All-Gather Matmul
-----------------

``all_gather_matmul`` keeps its architecture-based default routing when
``backend="auto"``. On SM100 and SM103, ``backend="cake"`` explicitly selects
the Cake fused backend for bfloat16 or float16 operands with ``K=8192``: a
contiguous ``[M, 8192]`` input with any positive ``M``, a ``[8192, N]`` weight
with ``N`` a positive multiple of 256 that is either contiguous or the
transposed view of a contiguous ``[N, 8192]`` parameter (each layout has its
own generated kernel; no copy is made), the process's torch symmetric-memory
backend being the default ``CUDA`` backend or ``NVSHMEM`` (the Cake backend
never selects one itself), and a two-, four- or eight-rank NCCL process group.
The local input may be an ordinary contiguous CUDA tensor because Cake uses
internal symmetric scratch and flags for remote access and synchronization;
the scratch of a group grows to the largest ``M`` seen, and that growth is the
only collective after the first call. Unsupported explicit Cake requests raise
instead of silently falling back. One generated source per kernel serves both
architectures; the JIT loader compiles it with the exact flag set of the
device it runs on.

``prepare_all_gather_matmul`` binds the weight, the process group and a row
capacity ``max_rows`` (default: the rows of the sample input) once, sizing the
symmetric scratch in that single collective, and returns a callable that
accepts any contiguous input with the same dtype, device and ``K`` and at
most ``max_rows`` rows. Both ``backend="auto"`` and ``backend="cake"`` select
this prepared Cake route; the same operand rules as the one-shot Cake route
apply. The returned launcher is CUDA-graph capturable: a captured call
allocates nothing but its output, and a call that would need a larger scratch
raises inside capture instead of running a collective. Unsupported
configurations raise during preparation instead of falling back.

.. autosummary::
    :toctree: ../generated

    all_gather_matmul
    prepare_all_gather_matmul

Fused FP8 Row-wise GEMM + All-Reduce
------------------------------------

``fp8_rowwise_gemm_allreduce`` computes, for a four-rank NCCL process group on
one SM100 node, ``Y = sum_r (A_r @ W_r.T) * (a_scale_r[:, None] * w_scale_r[None, :])``
with the per-rank FP8 (e4m3) operands ``A_r`` of shape ``[M, K_l]`` and ``W_r``
of shape ``[N, K_l]``, fp32 row and column scales applied before the cross-rank
sum, and a bfloat16 result ``[M, N]`` on every rank. ``backend="cake"`` (the
only backend; ``"auto"`` selects it) runs one fused kernel per call: the GEMM
epilogue stores each owned tile's bf16 partial into this rank's output and
publishes every other tile's partial into the chunk owner's symmetric receive
slot, the owner sums the four partials in fixed rank order and broadcasts the
result, so no separate all-reduce kernel or collective runs.
Operands are contiguous CUDA tensors with ``N`` a positive multiple of 256,
``K_l`` a positive multiple of 128, any positive ``M`` with ``M * N <= 2**30``,
and the process's torch symmetric-memory backend being the default ``CUDA``
backend or ``NVSHMEM`` (the Cake backend never selects one itself). The
symmetric workspace of a group grows to the largest ``M`` seen; that growth is
the only collective after the first call. Unsupported requests raise instead
of silently falling back.

``prepare_fp8_rowwise_gemm_allreduce`` binds the process group, ``N``, ``K_l``
and a row capacity ``max_rows`` (default: the rows of the sample input) once,
sizing the symmetric workspace in that single collective, and returns a
callable ``launcher(a, a_scale, w, w_scale, *, out=None)`` that accepts any
operands of the bound ``N`` and ``K_l`` with at most ``max_rows`` rows and
never runs another collective. The launcher is CUDA-graph capturable: a
captured call allocates nothing but its output and replays interleave with
eager calls; a call that exceeds the capacity raises before any launch.
``max_rows`` must be identical on every rank of the group: the preparation
checks it in that collective and raises ``ValueError`` on every rank when the
values differ.

.. autosummary::
    :toctree: ../generated

    fp8_rowwise_gemm_allreduce
    prepare_fp8_rowwise_gemm_allreduce

TensorRT-LLM AllReduce
----------------------

Types and Enums
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    AllReduceFusionOp
    AllReduceFusionPattern
    AllReduceStrategyConfig
    AllReduceStrategyType
    QuantizationSFLayout

Core Operations
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    trtllm_allreduce_fusion
    trtllm_moe_allreduce_fusion
    trtllm_moe_finalize_allreduce_fusion

Workspace Management
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    trtllm_create_ipc_workspace_for_all_reduce_fusion
    trtllm_destroy_ipc_workspace_for_all_reduce
    trtllm_destroy_ipc_workspace_for_all_reduce_fusion

Initialization and Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    trtllm_lamport_initialize
    trtllm_lamport_initialize_all
    compute_fp4_swizzled_layout_sf_size

Unified AllReduce Fusion API
----------------------------

.. autosummary::
    :toctree: ../generated

    allreduce_fusion
    create_allreduce_fusion_workspace
    AllReduceFusionWorkspace

All-reduce workspaces backed by ``SymmDeviceMemory`` preserve their CUDA
virtual addresses across process checkpoint/restore.  After quiescing all
work, release the physical handles and restore them with a fresh communication
backend before replaying a captured CUDA graph:

.. code-block:: python

    workspace.checkpoint_prepare()
    workspace.checkpoint_restore(comm_backend)

Both methods are collective.  Every rank must call them in the same order, and
``comm_backend`` must reproduce the original rank and world size.  Repeated
calls are no-ops after the workspace reaches the requested state.  If an
exception occurs after detach or reattach begins, do not retry or reuse the
workspace; restart the affected rank.  Workspaces backed by torch symmetric
memory do not support this lifecycle.

.. autoclass:: TRTLLMAllReduceFusionWorkspace
    :members:
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: MNNVLAllReduceFusionWorkspace
    :members:
    :show-inheritance:

    .. automethod:: __init__

MNNVL CuTe DSL backend
~~~~~~~~~~~~~~~~~~~~~~

``create_allreduce_fusion_workspace(backend="mnnvl-cutedsl", ...)`` builds an
:class:`~flashinfer.comm.mnnvl_cutedsl_ar.MNNVLCuteDSLAllReduceFusionWorkspace`:
CuTe DSL kernels for the ranks of one NVLink multicast domain, compiled for
one static problem. ``backend="auto"`` never selects it. The keyword-only
arguments of the factory (``top_k``, ``rms_eps``, ``weight_bias``,
``routed_scaling_factor``, ``include_shared_expert``, ``add_residual``,
``write_residual_output``, ``apply_rms_norm``, ``output_dtype``,
``dynamic_fp8``, ``write_norm_output`` and ``config``) are compile-time
settings: every :func:`allreduce_fusion` call on the workspace must use the
same epsilon, weight bias, residual, norm and output mode, and at most
``max_token_num`` tokens of ``hidden_dim`` BF16 elements. Other backends
reject these arguments. ``residual_in`` must be identical on every rank, as
under tensor parallelism: most kernels add the residual once, on the rank that
owns a token or a hidden-dimension shard.

The kernels spin on data written by the other ranks and some of them occupy
every SM. Do not run two of these collectives (for example from two
workspaces) concurrently on different streams: a rank whose SMs are all held
by one collective cannot start the other, which the peers may be waiting on.

The workspace picks a static profile for ``(world_size, hidden_dim, top_k,
dtype)`` and the device's compute capability, then routes each token count to
a low-latency (LL), balanced (BT) or high-throughput (HT) kernel. Construction
fails when no profile matches; :func:`is_mnnvl_cutedsl_supported` runs the same
checks without raising. It describes the local rank only, and construction is
collective, so agree on the result across ranks first:

.. code-block:: python

    ok = is_mnnvl_cutedsl_supported(
        4, 8192, torch.bfloat16, max_token_num=2048, apply_rms_norm=False
    )
    flag = torch.tensor(int(ok), device="cuda")
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    if flag.item():
        workspace = create_allreduce_fusion_workspace(
            backend="mnnvl-cutedsl",
            world_size=4,
            rank=rank,
            max_token_num=2048,
            hidden_dim=8192,
            dtype=torch.bfloat16,
            apply_rms_norm=False,
            add_residual=False,
        )
        out = allreduce_fusion(
            input=x, workspace=workspace, pattern=AllReduceFusionPattern.kAllReduce
        )

A workspace built with ``apply_rms_norm=False`` and ``add_residual=False``
serves ``kAllReduce``; the default norm-on workspace serves
``kARResidualRMSNorm`` and the MoE finalize pattern. An
``output_dtype=torch.float8_e4m3fn`` workspace serves the static FP8 patterns
(2 and 12, or 4 with ``write_norm_output=True``); one that also sets
``dynamic_fp8=True`` serves only the dynamic patterns (10, or 11 with
``write_norm_output=True``), not the static ones.

.. autosummary::
    :toctree: ../generated

    is_mnnvl_cutedsl_supported

.. autoclass:: flashinfer.comm.mnnvl_cutedsl_ar.MNNVLCuteDSLAllReduceFusionWorkspace
    :show-inheritance:

    .. automethod:: __init__

Cake NVFP4-Transport AllReduce
------------------------------

``cake_fp4_allreduce`` sums ``[rows, 6144]`` bfloat16 activations over a two-
or four-rank NCCL process group through NVFP4 records: every rank quantises
its rows (E2M1 values, one E4M3 scale per 16 values, one float32 row scale),
pushes them into Lamport receive windows of its peers and decodes the gathered
records in rank order with float32 accumulation, so the result is bitwise
identical on every rank.  World size 2 runs a one-shot exchange, world size 4
a two-shot exchange whose owner ranks reduce, re-encode and all-gather (one of
three phase-B schedules is chosen per call from the rows each CTA owns).
``cake_fp4_allreduce_nvfp4`` accepts pre-quantised ``[rows, 3456]`` records
with one static scale per rank.

A ``CakeFp4AllReduceWorkspace`` is created collectively on every rank of the
group with the same family (``"allreduce"`` or ``"rmsnorm"``), input format
(``"bf16"`` or ``"nvfp4"``) and row capacity; it shares the receive windows
through CUDA IPC, arms their sentinels and publishes the peer window table.
Calls accept any row count up to the capacity (a multiple of 4 for world size
4), perform no allocation when the output buffers are given and no host
synchronisation, so they can be captured into CUDA graphs; the
``launch_with_pdl`` flag issues the kernel as a programmatic dependent launch.
The programs are exact-architecture builds for SM100 (B200), SM103 (B300) and
SM107;
the build targets follow ``FLASHINFER_CUDA_ARCH_LIST`` when set, else the
visible devices.

``cake_fp4_allreduce_rmsnorm`` (family ``"rmsnorm"``, world size 2) fuses the
exchange with the residual add and a Gemma RMSNorm: ``z = sum + residual`` in
float32, ``residual_out = bf16(z)``, ``rinv = rsqrt(mean(z^2) + eps)`` from the
float32 sum of squares and ``y = bf16(z) * rinv * (weight_bias + weight)``,
with ``eps`` and ``weight_bias`` as runtime scalars of the
``CakeFp4AllReduceFusedNormSpec``.  The spec selects the output: bfloat16
``y``, or ``y`` quantised to NVFP4 (packed E2M1 values and 128x4-swizzled E4M3
block scales for a caller-supplied float32 global scale -- the layout and
arithmetic of ``fp4_quantize``) or to MXFP8 (E4M3 values and swizzled UE8M0
scales, as ``mxfp8_quantize``).  ``cake_fp4_allreduce_rmsnorm_nvfp4`` takes
pre-quantised records with the same epilogue.  Both return a
``CakeFp4AllReduceFusedNormResult``.

.. autosummary::
    :toctree: ../generated

    CakeFp4AllReduceWorkspace
    CakeFp4AllReduceFusedNormOutput
    CakeFp4AllReduceFusedNormSpec
    CakeFp4AllReduceFusedNormResult
    cake_fp4_allreduce
    cake_fp4_allreduce_nvfp4
    cake_fp4_allreduce_rmsnorm
    cake_fp4_allreduce_rmsnorm_nvfp4

FP8 Quantized AllReduce
-----------------------

.. currentmodule:: flashinfer.comm

.. autosummary::
    :toctree: ../generated

    quantized_all_reduce

vLLM AllReduce
--------------

.. autosummary::
    :toctree: ../generated

    vllm_all_reduce
    vllm_dispose
    vllm_init_custom_ar
    vllm_register_buffer
    vllm_register_graph_buffers
    vllm_get_graph_buffer_ipc_meta
    vllm_meta_size

PCIe IPC AllReduce
------------------

Custom all-reduce for intra-node PCIe machines without NVLink. Admission is a
capability check — world size, dtype, workspace capacity, and enough payload
for every rank to own a share — and shapes it rejects fall back to the caller's
own collective.

.. code-block:: python

    import flashinfer.comm as comm

    # Collective: every rank builds the workspace with identical arguments.
    # Size max_numel to the real workload -- an oversized one costs latency.
    ws = comm.PcieIpcAllReduceWorkspace(group=group, max_numel=128 * 6144)

    for x in activations:                     # same shapes, same order, all ranks
        if ws.supports(x):
            y = ws.all_reduce(x)              # out-of-place
        else:
            y = x.clone()
            dist.all_reduce(y, group=group)   # unsupported shape: fall back

    ws.destroy()                              # collective; all ranks together

``supports()`` is a pure function of shape and dtype, so every rank reaches the
same answer without agreeing on one at runtime. It says nothing about speed: a
supported shape runs a seed launch configuration — one crossover keyed on the
payload in bytes, no per-machine constants — and warns once per workspace until
:meth:`~PcieIpcAllReduceWorkspace.tune` has measured the real one and persisted
it, since the crossovers depend on the fabric.
:func:`get_pcie_ipc_launch_config` exposes that seed for a
``(world_size, numel, elem_size)`` triple, and returns ``None`` only for shapes
the kernels cannot run.

The kernels spin on peer flags with no timeout, so ranks that disagree on
shape, dtype or call order hang rather than raise. One workspace serves one
CUDA stream; use :meth:`~PcieIpcAllReduceWorkspace.rebind_stream` after
ordering the two if a move is genuinely needed.

Two data planes
~~~~~~~~~~~~~~~

Above roughly a megabyte the workspace can move the payload with the copy
engines instead of with SM load/store. The reason is not that the copy engine
is faster per hop — in isolation the two are comparable — but that it keeps
its throughput when every rank transfers at once, where SM peer traffic loses
bandwidth to the concurrency. So the gap widens with world size, which is what
makes a second plane worth having at these payload sizes.

It is a second protocol rather than a flag on the existing kernels. The SM path
encodes readiness in the payload — ``+0.0`` means "not yet written" — which a
copy engine can neither produce nor observe, so the ring synchronises through
monotonic flags on side streams instead. Nothing about the SM kernels changes.

Both planes are ordinary tuner candidates, so nothing needs to be declared at
the call site and there is no notion of a prefill or decode "phase": the
configuration follows the payload in bytes.
:class:`PcieIpcVariant` gains ``COPY_ENGINE_RING``, a flat neighbour ring, and
``COPY_ENGINE_ISLAND``, a 4+4 decomposition offered only at world size 8 and
only on ``rootcplx-noswitch``, where a ring's two socket-crossing hops would
otherwise set the pace. On that variant ``blocks`` carries the ring's sub-chunk
depth rather than a grid size.

.. warning::

    Tuning covers the batch sizes it is given, and shapes above the largest one
    are served by whatever was measured there. Leave ``tune_batches`` unset and
    the ladder is derived from ``max_numel``, which covers everything the
    workspace admits; pass it explicitly and the top of the range is yours to
    get right. :meth:`~PcieIpcAllReduceWorkspace.tune` warns when an explicit
    list leaves a gap, and refuses to persist results measured below the
    default sample counts, since a table that is quietly wrong is worse than
    one that is missing.

.. autosummary::
    :toctree: ../generated

    PcieIpcAllReduceWorkspace
    PcieIpcLaunchConfig
    PcieIpcVariant
    get_pcie_ipc_launch_config
    probe_pcie_ipc_rank_topology
    resolve_pcie_ipc_profile

PCIe IPC AllGather and ReduceScatter
------------------------------------

These standalone workspaces support BF16, FP16, and FP32 tensors at world sizes
2, 4, and 8. ``max_numel`` sizes one rank's shard: the AllGather input or the
ReduceScatter output. Tensors must be contiguous, rank-2, 16-byte aligned, and
contain complete 16-byte packs.

Their Python implementation lives in ``flashinfer.comm.pcie_ipc_collectives``;
the public entry points remain the ``flashinfer.comm`` exports shown below.

As with :class:`PcieIpcAllReduceWorkspace`, construction, calls, and destruction
are collective. Every rank must issue the same sequence and configuration, and
one workspace belongs to one ordered CUDA stream. Keep the workspace alive
until every captured CUDA graph replay has completed.

AllGather copies opaque 16-byte packs. ReduceScatter accumulates in FP32 and
converts to the requested output dtype; its TP8 schedule converts one four-rank
partial before the final accumulation, so bitwise agreement with NCCL is not
part of the contract.

The TP8 CopyEngine AllGather and topology ReduceScatter schedules require live
UUID/NVML evidence that logical ranks ``0..3`` and ``4..7`` form the expected
two-island placement. AllGather falls back to recursive doubling when that
proof is unavailable; TP8 ReduceScatter reports the input as unsupported.

.. code-block:: python

    import flashinfer.comm as comm

    ag = comm.PcieIpcAllGatherWorkspace(group, max_local_rows * hidden, dtype=x.dtype)
    rs = comm.PcieIpcReduceScatterWorkspace(group, max_local_rows * hidden, dtype=x.dtype)

    gathered = ag.all_gather(x)            # [world_size * local_rows, hidden]
    shard = rs.reduce_scatter(reduced_input)  # [local_rows, hidden]

    ag.destroy()
    rs.destroy()

``launch_config()`` returns a conservative seed. ``tune([hidden, ...])`` checks
every candidate against NCCL, minimizes the maximum latency across ranks, and
persists exact-shape results. The cache key includes the collective, world size,
topology/rank-placement fingerprint, workspace limits, dtype, and shape.

The seed thresholds are heuristics; support does not imply a performance win.
The variant names combine communication schedule, transport, and batching:
``COPY_ENGINE`` uses recursive doubling, and ``ONE_PACK`` means one 16-byte
pack per thread in a grid-sized batch. See
:doc:`PCIe IPC AG/RS schedules <../design_docs/pcie_ipc_ag_rs>` for the seven
variants, synchronization/traffic models, topology boundary, and measurement
provenance.

.. autosummary::
    :toctree: ../generated

    PcieIpcAllGatherWorkspace
    PcieIpcAllGatherLaunchConfig
    PcieIpcAllGatherVariant
    get_pcie_ipc_all_gather_launch_config
    PcieIpcReduceScatterWorkspace
    PcieIpcReduceScatterLaunchConfig
    PcieIpcReduceScatterVariant
    get_pcie_ipc_reduce_scatter_launch_config

Ulysses Context-Parallel All-to-All
-----------------------------------

.. currentmodule:: flashinfer.comm

Communication for Ulysses context parallelism over the 4-D layout
``[B, S, H, D]``. Two layout transforms are provided; a typical attention
layer makes four collective calls (q/k/v through ``scatter_heads``, the
output through ``gather_heads``):

- ``scatter_heads``: ``[B, S_local, H, D] -> [B, S_global, H_local, D]`` —
  each rank keeps head slice ``[rank * H_local, (rank+1) * H_local)`` of the
  *full* sequence;
- ``gather_heads``: ``[B, S_global, H_local, D] -> [B, S_local, H, D]`` —
  the inverse, returning all heads of this rank's sequence shard,

with ``H_local = H // world_size`` and ``S_global = S_local * world_size``.
All backends produce bit-identical results.

A third entry point, ``exchange_chunks``, takes ``[1, 1, world_size, chunk]``
to the same shape: chunk ``r`` goes to peer ``r`` and lands in slot ``rank``
there, which is ``dist.all_to_all_single`` over an already-packed payload. It
interleaves nothing, so it is for operands a preceding pack (a quantizer, say)
already laid out destination-major — the head-axis transforms would make such a
producer emit head-major and repack. On the ``"pcie"`` RDMA routes it is also
the cheaper descriptor: each peer's bytes are contiguous on both ends, so the
NIC addresses them with the plain memory region and a linear offset instead of
an interleaved MKey, which is what exempts it from the 65535-byte stride limit
that bounds ``H*D*element_size`` for the other two.

**Backend policy.** :class:`UlyssesCommunicator` selects its backend in the
constructor, strictly before any IPC allocation or JIT compilation:

============== ==================================================================
``backend=``   behavior
============== ==================================================================
``"auto"``     fused-transpose NVLink-P2P kernel when the group is a verified
               single-node all-pairs NVLink mesh with a supported world size
               (2/4/6/8); the experimental PCIe transport where NVLink is
               unavailable and ``FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1``
               is set on every rank; NCCL otherwise. The instance exposes ``.backend``
               (effective), ``.fallback_reason`` and ``.decision`` /
               ``.topology_decision``.
``"nvlink"``   force the fused kernel; raises on every rank (before IPC/JIT for
               topology failures) when it cannot be used.
``"pcie"``     force the experimental single-node PCIe transport at world size
               1/2/4/8. Size 1 is an identity path and does not JIT-compile or
               arm a native transport. Size 2 uses CUDA P2P. Sizes 4/8 prefer
               an all-RDMA route (per-rank mlx5 to every peer) and fall
               back to all-P2P; ``FLASHINFER_ULYSSES_PCIE_ROUTE`` can force
               all-P2P, all-RDMA at any multi-rank world size, or the
               eight-rank 4+4 NUMA P2P/mlx5 hybrid. Every route requires
               full-group CUDA P2P access for the epoch signal barrier.
               This backend is experimental: ``auto`` considers it only
               where NVLink is unavailable and
               ``FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`` is set on
               every rank. Naming it explicitly never needs that variable.
               Either way it warns once.
``"nccl"``     force ``dist.all_to_all_single`` + permute; skips the
               topology/NVML probe and all IPC/JIT (the constructor still
               resolves/guards the CUDA device and performs CUDA-backed
               metadata collectives); any world size.
============== ==================================================================

Typical fallback reasons reported by ``.fallback_reason`` (all conservative —
anything unknown or unverifiable selects NCCL): unsupported world size (only
2/4/6/8 have fused-kernel instantiations; ``world_size == 1`` is a no-copy
passthrough), ranks spanning multiple hosts, missing pair-wise P2P or NVLink
between any two concrete GPUs (verified per pair via NVML, not "some active
link"), duplicate or unknown physical GPU identity, a topology probe error,
inconsistent per-rank decisions, or a runtime NVLink initialization failure
after a positive topology decision.

**Constraints.** The constructor is always collective (all ranks together);
:meth:`UlyssesCommunicator.close` is collective when the NVLink or PCIe backend
was armed — for the pure NCCL backend, ``world_size == 1``, or an auto
fallback whose NVLink cleanup already completed, ``close`` is local and
idempotent. Rank-local failures inside backend initialization or a
collective ``close`` are exchanged as group outcomes so all ranks jointly
clean up and raise (or fall back) instead of deadlocking, and a failed
``close`` may be retried. All ranks must request the same ``backend`` and
agree on ``max_bytes`` and ``dtype``; each rank may bind a different CUDA
device (``device`` accepts ``torch.device``, ``str`` or an ``int`` ordinal,
e.g. ``cuda:rank``). With ``world_size > 1`` the NCCL and NVLink backends
(NVLink's ``exchange_chunks`` runs ``all_to_all_single`` too) require
``group`` to support CUDA all-to-all (an NCCL process group), checked at
construction before any resource is armed. Operands must be contiguous 4-D
CUDA tensors of the construction ``dtype`` (float16 / bfloat16 / float32
only) on the construction device, every dim positive, and ``nbytes`` at most
``max_bytes``; the element count of any one operand must additionally stay
within the int32 kernel index range. ``scatter_heads`` requires
``H % world_size == 0``
and ``gather_heads`` requires ``S_global % world_size == 0``. Collectives
run on the current CUDA stream; all ranks must issue the same call sequence
with consistent shapes, one collective in flight per communicator at a
time.

**Destination-passing and workspace mode.** Both collectives accept an
optional preallocated ``out=`` tensor. The NCCL backend additionally accepts
a reusable :class:`UlyssesWorkspace`, which owns the packed send and receive
buffers and removes their per-call allocations. Omitting both keywords keeps
the original allocation path unchanged. The NVLink backend uses ``out=`` but
does not consume the workspace because its IPC staging allocation is owned by
the communicator.

**Registered NVLink outputs.** On the NVLink backend's Cake route (SM100 /
SM103, ``nvlink_kernel == "cake"``), :meth:`UlyssesCommunicator.allocate_output`
is collective, as on the PCIe backend: every rank allocates ``max_bytes`` bytes
and maps every peer's allocation over CUDA IPC. A ``scatter_heads`` or
``gather_heads`` whose ``out=`` is such an output writes each rank's blocks
straight into the destination ranks' outputs, with no pass through the
staging buffer. All ranks pass their output of the same registration;
outputs stay registered until ``close``, which frees them. Caller-owned
outputs keep the staged path, and the two may alternate on one communicator.
Call ``allocate_output`` for every geometry before CUDA graph capture.

**PCIe backend (experimental).** It additionally accepts the 1-, 2- and
4-byte element types float16 / bfloat16 / float32, int8 / uint8 and
float8_e4m3fn / float8_e5m2; its mlx5 routes — the eight-rank 4+4 NUMA
hybrid and the ``FLASHINFER_ULYSSES_PCIE_ROUTE=rdma`` all-RDMA route —
require batch size 1 and a head-row pitch ``H * D * element_size <= 65,535``
bytes and refuse CUDA graph capture, while the all-P2P route takes any batch
size and can be captured when every output comes from
:meth:`UlyssesCommunicator.allocate_output`. The JIT module links
libibverbs, libmlx5 and the CUDA driver library even when topology selects
the all-P2P route, so the rdma-core development packages must be present;
it is the only FlashInfer JIT module that links them. Multi-rank PCIe calls
require ``out=`` from :meth:`UlyssesCommunicator.allocate_output`; the
backend has no staging buffers, so ``workspace=``,
:meth:`UlyssesCommunicator.create_workspace` and the head-chunk primitives
are refused at world size above one. A rejection before native code runs
(a bad operand, a refused capture, a second stream) leaves the RDMA routes
usable, since every rank rejects the same call under the SPMD contract; the
all-P2P route, whose barrier has no abort protocol, treats it as fail-stop.
A failure inside the transport is fail-stop on every route: the communicator
enters a BROKEN state that permits only a collective ``close``. On the RDMA
routes it publishes a sticky abort that releases the peers and ``close``
completes; on all-P2P ``close`` refuses to synchronize and the process must
exit.
See the `PCIe Ulysses design note
<https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/ulysses_pcie.md>`_
for the support matrix, output-lifetime rules, teardown protocol, routing
controls, and benchmark.

**Known limitations.**

- PyTorch builds without ``torch.cuda.get_device_properties(...).uuid``
  cannot establish physical GPU identity: ``auto`` conservatively falls back
  to NCCL (the reason names the missing attribute).
- When each process can only see its own GPU (e.g. one
  ``CUDA_VISIBLE_DEVICES`` entry per rank), peers are invisible to the P2P
  probe and ``auto`` falls back to NCCL.
- Out-of-range CUDA ordinals passed as *strings or ints* are rejected at
  construction; a pre-built ``torch.device`` object wraps its index into a
  signed byte before FlashInfer can see it (``torch.device("cuda:256")`` is
  already ``cuda:0``), so only the surviving index can be range-checked.
- Teardown metadata exchanges run bound to the communicator device; an
  extreme failure in the guard *restore* path after a completed collective
  can still desynchronize ranks (never observed in tests; tracked as a
  hardening note).

**Example** (Wan2.1-style attention; see the
`wan example <https://github.com/flashinfer-ai/flashinfer/tree/main/examples/pytorch/wan>`_
for the full integration)::

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    with UlyssesCommunicator(group, max_bytes=q.nbytes,
                             dtype=torch.bfloat16, device=device) as comm:
        q_ = comm.scatter_heads(q)   # [B,S_local,H,D] -> [B,S_global,H_local,D]
        k_ = comm.scatter_heads(k)
        v_ = comm.scatter_heads(v)
        o_ = attention(q_, k_, v_)
        o = comm.gather_heads(o_)    # [B,S_global,H_local,D] -> [B,S_local,H,D]

Preallocated outputs and NCCL staging can be reused across serialized calls::

    workspace = comm.create_workspace()
    q_global = torch.empty(B, S_global, H_local, D,
                           dtype=q.dtype, device=q.device)
    comm.scatter_heads(q, out=q_global, workspace=workspace)

.. autoclass:: UlyssesCommunicator
    :members:
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: UlyssesWorkspace
    :members:
    :show-inheritance:

    .. automethod:: __init__

Head-Chunk Layout and Transport Primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Head-chunk APIs are opt-in building blocks for framework-owned
communication/compute pipelines. They do not choose a chunk schedule, create
streams/events or process groups, or alter the ordinary ``scatter_heads`` /
``gather_heads`` path.

``pack_ulysses_qkv_head_chunk`` accepts independent positive-strided Q/K/V
views, selects the same destination-local head band from every Ulysses rank's
head slice, and fuses Q/K/V into the last dimension. The communicator's
``scatter_qkv_head_chunk`` method combines that transform with one all-to-all.
After attention, ``gather_output_head_chunk`` performs the reverse all-to-all
and writes the compact band directly into the full output. A framework that
overlaps the two communication directions must use separate communicators and
workspaces::

    schedule = ((0, 3), (3, 8), (11, 3))  # local_heads == 14
    full_output = torch.empty_like(q)
    for head_offset, head_count in schedule:
        qkv = input_comm.scatter_qkv_head_chunk(
            q, k, v,
            head_offset=head_offset,
            head_count=head_count,
            out=qkv_chunk_buffers[head_count],
            workspace=input_workspace,
        )
        q_chunk, k_chunk, v_chunk = qkv.split(head_dim, dim=-1)
        o_chunk = attention(q_chunk, k_chunk, v_chunk)
        output_comm.gather_output_head_chunk(
            o_chunk,
            local_heads=14,
            head_offset=head_offset,
            out=full_output,
            workspace=output_workspace,
        )

Head chunking is not automatically profitable. In particular, tensor
parallelism reduces the effective local head count to approximately
``H / (TP * Ulysses)``. Frameworks should enable a schedule only after all
ranks agree that the attention backend supports every chunk head count and an
offline benchmark shows a positive result. Small local-head counts, GQA/MLA
with unequal Q/K/V heads, Ring Attention, and Attention2D should use the
ordinary path unless separately implemented and measured.

.. autofunction:: pack_ulysses_qkv_head_chunk

.. autofunction:: merge_ulysses_output_head_chunk

Fused QKV Projection + Norm + RoPE + Head Exchange
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`QkvNormRopeExchangeOverlap` fuses the attention front end of a
sequence-parallel (Ulysses) layer into one prepared session: the BF16 QKV
projection, per-head RMSNorm of Q and K (one ``[D]`` scale each), interleaved
RoPE from a per-token ``[rows, D]`` cos/sin table, and the head exchange.  The
Cake backend (``backend="nvlink"``; SM90, SM100, SM103 and SM107, single-node NVLink
groups of 2, 4 or 8 ranks (2 or 4 on SM107), ``head_dim=128``, BF16) stores every destination's
head band straight from the GEMM epilogue into that rank's symmetric receive
slot and publishes per-chunk readiness words, so the exchange overlaps the
projection instead of following it; a one-CTA wait kernel on a consumer stream
acquires the words and the caller's stream joins it.  The outputs are views of
the current receive slot (``layout="bshd"``: ``(q, k, v)`` each
``[B, S_global, H_local, D]``; ``layout="fused"``: one
``[B, S_global, H_local, 3 * D]`` payload) and stay valid until the slot is
reused ``num_slots`` calls later; pass ``out=`` for an owned copy.

Per call the Cake backend launches, by default on every architecture, the
producer build with the folded cross-rank rendezvous (a rendezvous warp of
CTA 0 runs the NVLink barrier of the call's phase and releases the stores
through a rank-local arrive flag: one extra warp per CTA on SM100 / SM103, the
idle TMA warp on SM90) on the caller's stream and the wait kernel on the
consumer stream: two launches.  ``fold_barrier=False`` (or the environment
variable ``FLASHINFER_QKV_EXCHANGE_FOLD_BARRIER=0``) selects the three-launch
path instead: the barrier kernel, the plain producer and the wait kernel; both
paths are delivered on every architecture.  Both paths store identical bits;
``describe()`` reports the resolved path (``fold_barrier``) and its launch
count (``launches_per_call``).

The producer build is selected per architecture and world size
(``cake_qkv_norm_rope_exchange_overlap.PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD``
of the JIT loader): on SM100 a group of 8 ranks runs the build with wide
operand loads and bulk row stores (``bulk_ow256``), a group of 4 ranks the
build with wide operand loads (``ow256``); on SM103 groups of 8 and 4 ranks run
``ow256``; on SM107 a group of 4 ranks runs the build with bulk row stores
(``bulk``); groups of 2 ranks and SM90 run the baseline build.  Every build
stores identical bits; the wide-load builds require 32-byte-aligned ``cos_sin``
and norm-weight tensors (a misaligned view raises and names the opt-out).  The
environment variable ``FLASHINFER_QKV_EXCHANGE_PRODUCER_VARIANT`` (``auto`` /
``shipped`` / a delivered build name) overrides the selection per session,
rank-uniformly; the baseline build stays delivered beside every variant, and a
build that is not delivered for the session's tile width raises and names the
delivered ones.  The host plan's group scaling follows the same keying
(``cake_qkv_exchange_plan.GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD``): on SM100
groups of 4 and 2 ranks and on SM103 every group size double the token-tile
group of a grouped row whose hidden size is at least 6144 (the tile order
changes, every destination element is still written once); SM100 groups of 8
ranks, SM90 and SM107 keep the baseline plan.  ``FLASHINFER_QKV_EXCHANGE_GROUP_M_SCALE``
(``auto`` / ``1`` / ``2`` / ``4``) overrides it per session, rank-uniformly.

Per-architecture row-count gates shape each call's host plan: on SM100 and
SM103 a call with at most 4096 rows per rank publishes one readiness word per
destination, and a call with at least 16384 rows per rank launches the producer
once per destination run of its tile table (the first launch carrying the
folded rendezvous); on SM90 a call with at most 4096 rows per rank publishes one
readiness word per destination, a call with at least 16384 rows per rank runs the
persistent producer grid over every SM (no reserved SMs), and the per-destination
launch gate is off; on SM107 every gate is off.  One cell differs from these
per-architecture tables (``cake_qkv_exchange_plan.SHAPE_GATE_DEFAULT_BY_ARCH_WORLD``):
a group of 8 ranks on SM90 publishes one readiness word per destination for
every row count (``words_capacity_small_m = all``).  A fourth gate,
``pair_off_large_m``, is off on every architecture: switched on
(``pair_off_large_m=N``) it runs a 256-column call with at least ``N`` rows per
rank on the single-CTA producer instead of the 2-CTA pair tile (the tile order
is unchanged); it needs that producer delivered for the architecture -- no
delivered Blackwell route ships it today, so the session refuses at
construction when a 256-column call can reach the gate (``max_tokens >= N``),
naming the gate -- and is refused on SM90, which builds no pair tile.  The
environment variable ``FLASHINFER_QKV_EXCHANGE_SHAPE_GATES`` (``auto`` / ``off`` /
``words_capacity_small_m=N|all,reserved_sms_large_m=N,per_dest_large_m=N,pair_off_large_m=N``)
overrides them per session, rank-uniformly (``words_capacity_small_m=4096`` on
SM90 with 8 ranks restores the previous plan); ``describe()`` reports the resolved
build (``producer_variant``), scale (``group_m_scale``) and gates
(``shape_gates``) with their sources, and every prepared row's verdict
(``prepared_row_plans``, including its ``reserved_sms``).  The first three gates
change the host plan only -- neither the tile table nor the stored bits; the
pair-tile gate changes the producer program and the tile table (no lockstep
pads) of the rows it selects; no gate changes the stored bits.  The largest
2-rank shape of the evaluation grid (``S_global = 131072``, ``hidden = 8192``)
is not supported on an 80 GB H100: its operands and receive slots exceed the
device memory.

The session is prepared eagerly (symmetric slots, readiness pad, barrier
flags, JIT programs, the tile plan of ``max_tokens`` rows) and a prepared call
is CUDA-graph capturable: a captured call allocates nothing, runs no
collective, publishes a sentinel epoch and clears its readiness words with a
stream memset after the wait kernel, and replays interleave with eager calls.
A call with more rows than ``max_tokens``, or with a row count first seen
inside a capture, raises instead of allocating.  ``backend="auto"`` selects
the Cake path when :func:`resolve_ulysses_backend` verifies the NVLink mesh and
the device is a supported architecture, and otherwise the NCCL path (cuBLAS
projection, :func:`flashinfer.norm.rmsnorm`, RoPE, then
:meth:`UlyssesCommunicator.scatter_qkv_head_chunk`), which is not graph
capturable (``graph_capturable`` reports ``False``).

.. autosummary::
    :toctree: ../generated

    QkvNormRopeExchangeOverlap

.. autoclass:: QkvNormRopeExchangeOverlap
    :members:
    :show-inheritance:

    .. automethod:: __init__

Topology Probing and Backend Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    UlyssesBackendDecision
    UlyssesRankTopology
    UlyssesPciePlan
    UlyssesBackendError

The communicator exposes ``.backend`` (the effective backend), ``.transport``
(``"p2p"``, ``"hybrid"`` or ``"rdma"``, PCIe only), ``.decision`` /
``.topology_decision`` and ``.fallback_reason``.

.. autofunction:: resolve_ulysses_backend

.. autofunction:: decide_ulysses_backend

.. autofunction:: probe_ulysses_rank_topology

Raw Kernel Entry Points (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prefer :class:`UlyssesCommunicator`; these assume the caller has already
verified all-pairs NVLink P2P and owns the IPC workspace lifecycle.

.. autosummary::
    :toctree: ../generated

    init_ulysses_a2a
    dispose_ulysses_a2a
    ulysses_a2a

MNNVL (Multi-Node NVLink)
-------------------------

``MnnvlMemory`` and ``SymmDeviceMemory`` allocate the symmetric memory behind
the NVLink collectives. The `symmetric memory design notes
<https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/symmetric_memory.md>`_
describe how each lays out its memory and how the ranks choose between FABRIC
handles and POSIX file descriptors to share it.

.. currentmodule:: flashinfer.comm.mnnvl

Core Classes
~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    MnnvlMemory
    McastGPUBuffer

TensorRT-LLM MNNVL AllReduce
----------------------------

.. currentmodule:: flashinfer.comm.trtllm_mnnvl_ar

.. autosummary::
    :toctree: ../generated

    trtllm_mnnvl_all_reduce
    trtllm_mnnvl_allreduce
    trtllm_mnnvl_fused_allreduce_add_rmsnorm
    trtllm_mnnvl_fused_allreduce_add_rmsnorm_quant
    trtllm_mnnvl_fused_allreduce_rmsnorm
    mpi_barrier

DCP All-to-All (Context-Parallel Attention Reduction)
-----------------------------------------------------

.. currentmodule:: flashinfer.comm

.. autosummary::
    :toctree: ../generated

    decode_cp_a2a_workspace_size
    decode_cp_a2a_allocate_mnnvl_workspace
    decode_cp_a2a_init_workspace
    decode_cp_a2a_alltoall

NCCL LSA DCP All-to-All + LSE Reduce
-------------------------------------

The fused path uses PyTorch NCCL symmetric memory and requires every context-
parallel rank to be in one load/store-accessible NVLink domain. It does not use
the MNNVL workspace accepted by ``decode_cp_a2a_alltoall``.

.. autosummary::
    :toctree: ../generated

    decode_cp_a2a_lse_reduce_workspace_size
    decode_cp_a2a_lse_reduce_create_workspace
    decode_cp_a2a_lse_reduce

TP MoE Finalize + RMSNorm
-------------------------

Tensor-parallel MoE finalize fused with the all-reduce and per-expert RMSNorm,
over an NVLink multicast workspace (hidden size 4096, top-k 8). Initialize the
process-wide workspace collectively with ``init_tp_moe_finalize_rmsnorm`` before
the first call.

.. currentmodule:: flashinfer.comm

.. autosummary::
    :toctree: ../generated

    init_tp_moe_finalize_rmsnorm
    supports_tp_moe_finalize_rmsnorm
    tp_moe_finalize_rmsnorm

MNNVL CuTe DSL Fused Collectives
--------------------------------

.. currentmodule:: flashinfer.comm.mnnvl_cutedsl_moe_finalize

.. autosummary::
    :toctree: ../generated

    MNNVLCuteDSLMoEFinalizeAllReduceWorkspace
    supports_moe_finalize_rmsnorm_allreduce
    moe_finalize_rmsnorm_allreduce

.. currentmodule:: flashinfer.comm.mnnvl_cutedsl_ag

.. autosummary::
    :toctree: ../generated

    MNNVLCuteDSLAllGatherDualNormWorkspace
    supports_all_gather_gated_residual_next_rmsnorm
    all_gather_gated_residual_next_rmsnorm
    MNNVLCuteDSLAllGatherNormQuantWorkspace
    supports_all_gather_rmsnorm_fp4_quantize
    all_gather_rmsnorm_fp4_quantize

Mixed Communication
-------------------

.. currentmodule:: flashinfer.comm.mixed_comm

.. autosummary::
    :toctree: ../generated

    MixedCommOp
    MixedCommMode
    MixedCommHandler
    run_mixed_comm
