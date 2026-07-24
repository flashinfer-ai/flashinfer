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
    trtllm_custom_all_reduce
    trtllm_moe_allreduce_fusion
    trtllm_moe_finalize_allreduce_fusion

Workspace Management
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    trtllm_create_ipc_workspace_for_all_reduce
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
Both backends produce bit-identical results.

**Backend policy.** :class:`UlyssesCommunicator` selects its backend in the
constructor, strictly before any IPC allocation or JIT compilation:

============== ==================================================================
``backend=``   behavior
============== ==================================================================
``"auto"``     fused-transpose NVLink-P2P kernel when the group is a verified
               single-node all-pairs NVLink mesh with a supported world size
               (2/4/6/8); NCCL otherwise. The instance exposes ``.backend``
               (effective), ``.fallback_reason`` and ``.decision`` /
               ``.topology_decision``.
``"nvlink"``   force the fused kernel; raises on every rank (before IPC/JIT for
               topology failures) when it cannot be used.
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
:meth:`UlyssesCommunicator.close` is collective only when the NVLink backend
was armed — for the pure NCCL backend, ``world_size == 1``, or an auto
fallback whose NVLink cleanup already completed, ``close`` is local and
idempotent. Rank-local failures inside the NVLink initialization or a
collective ``close`` are exchanged as group outcomes so all ranks jointly
clean up and raise (or fall back) instead of deadlocking, and a failed
``close`` may be retried. All ranks must request the same ``backend`` and
agree on ``max_elems`` and ``dtype``; each rank may bind a different CUDA
device (``device`` accepts ``torch.device``, ``str`` or an ``int`` ordinal,
e.g. ``cuda:rank``). With ``world_size > 1`` the NCCL backend (forced or
fallen back to) requires ``group`` to support CUDA all-to-all (an NCCL
process group), checked at construction. Operands must be contiguous 4-D
CUDA tensors of the construction ``dtype`` (float16 / bfloat16 / float32
only) on the construction device, every dim positive, at most ``max_elems``
(≤ 2^31 − 1) elements; ``scatter_heads`` requires ``H % world_size == 0``
and ``gather_heads`` requires ``S_global % world_size == 0``. Collectives
run on the current CUDA stream; all ranks must issue the same call sequence
with consistent shapes, one collective in flight per communicator at a
time.

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

    with UlyssesCommunicator(group, max_elems=B * S_local * H * D,
                             dtype=torch.bfloat16) as comm:
        q_ = comm.scatter_heads(q)   # [B,S_local,H,D] -> [B,S_global,H_local,D]
        k_ = comm.scatter_heads(k)
        v_ = comm.scatter_heads(v)
        o_ = attention(q_, k_, v_)
        o = comm.gather_heads(o_)    # [B,S_global,H_local,D] -> [B,S_local,H,D]

.. autoclass:: UlyssesCommunicator
    :members:
    :show-inheritance:

    .. automethod:: __init__

Topology Probing and Backend Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated

    resolve_ulysses_backend
    decide_ulysses_backend
    probe_ulysses_rank_topology
    UlyssesBackendDecision
    UlyssesRankTopology
    UlyssesBackendError

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

MNNVL A2A (Throughput Backend)
-------------------------------

.. currentmodule:: flashinfer.comm

.. autosummary::
    :toctree: ../generated

    moe_a2a_initialize
    moe_a2a_dispatch
    moe_a2a_combine
    moe_a2a_sanitize_expert_ids
    moe_a2a_get_workspace_size_per_rank
    moe_a2a_wrap_payload_tensor_in_workspace

.. autoclass:: MoeAlltoAll
    :members:
    :inherited-members:
    :show-inheritance:

    .. automethod:: __init__

``MoeAlltoAll`` preserves its CUDA virtual addresses across process
checkpoint/restore.  After quiescing all work, call ``checkpoint_prepare`` to
release the non-checkpointable physical MNNVL handles.  Then call
``checkpoint_restore`` with a fresh communication backend before replaying a
captured CUDA graph:

.. code-block:: python

    moe_alltoall.checkpoint_prepare()
    moe_alltoall.checkpoint_restore(comm_backend)

Both methods are collective.  Every rank must call them in the same order, and
``comm_backend`` must reproduce the original rank and world size.
Repeated calls are no-ops after the workspace reaches the requested state.
If an exception occurs after physical handle unmapping or remapping begins,
do not retry or reuse the workspace; restart the affected rank.

.. autosummary::
    :toctree: ../generated

    MoeAlltoAll.checkpoint_prepare
    MoeAlltoAll.checkpoint_restore

DCP All-to-All (Context-Parallel Attention Reduction)
-----------------------------------------------------

.. currentmodule:: flashinfer.comm

.. autosummary::
    :toctree: ../generated

    decode_cp_a2a_workspace_size
    decode_cp_a2a_allocate_mnnvl_workspace
    decode_cp_a2a_init_workspace
    decode_cp_a2a_alltoall

Mixed Communication
-------------------

.. currentmodule:: flashinfer.comm.mixed_comm

.. autosummary::
    :toctree: ../generated

    MixedCommOp
    MixedCommMode
    MixedCommHandler
    run_mixed_comm
