.. _moe-ep-api:

flashinfer.moe_ep
=================

.. currentmodule:: flashinfer.moe_ep

``flashinfer.moe_ep`` provides expert parallelism (EP) for MoE layers. It
supports two execution modes:

* **Split**: dispatch tokens to the owning ranks, run a local expert kernel,
  and combine the results. NCCL-EP and NIXL-EP are transport options.
* **Mega**: use one fused symmetric-memory kernel for dispatch, expert compute,
  and combine.

The public entry point is :func:`MoEEpLayer`. It returns a split or mega layer
based on the ``backend`` configuration. Both modes use the same
:class:`MoEEpTensors` input bundle and :class:`MoEWeightPack` weight
container.

The EP world is the process group described by :class:`BootstrapConfig`.
``world_size`` and ``rank`` must refer to that EP group, not necessarily the
default process group. In a host framework such as vLLM, pass the EP process
group explicitly when it differs from ``WORLD``.

Quick start
-----------

The following example shows the common split path with NCCL-EP and a local
identity kernel. Initialize the process group before running this code and
launch one process per EP rank. Replace ``IdentityConfig`` with a supported
local MoE kernel configuration for production use.

.. code-block:: python

   import torch
   import torch.distributed as dist

   from flashinfer.moe_ep import (
       BootstrapConfig,
       FleetParams,
       IdentityConfig,
       MoEEpLayer,
       MoEEpTensors,
       SplitConfig,
       dummy_moe_weights,
   )

   rank = dist.get_rank()
   world_size = dist.get_world_size()
   device = torch.device("cuda", torch.cuda.current_device())
   num_tokens = 32
   hidden_size = 4096
   num_experts = 8 * world_size
   top_k = 2
   hidden_states = torch.randn(
       num_tokens, hidden_size, device=device, dtype=torch.bfloat16
   )
   topk_ids = torch.randint(
       num_experts, (num_tokens, top_k), device=device, dtype=torch.int64
   )
   topk_weights = torch.softmax(
       torch.randn(num_tokens, top_k, device=device), dim=-1
   )

   bootstrap = BootstrapConfig(
       world_size=world_size,
       rank=rank,
       device=device.index,
   )
   fleet_params = FleetParams(
       num_experts=num_experts,
       max_tokens_per_rank=1024,
       token_hidden_size=hidden_size,
   )
   weights = dummy_moe_weights(
       num_local_experts=num_experts // world_size,
       hidden=hidden_size,
       device=device,
   )
   layer = MoEEpLayer(
       bootstrap,
       fleet_params,
       weights,
       backend=SplitConfig(kernel=IdentityConfig()),
   )
   tensors = MoEEpTensors(
       hidden_states=hidden_states,
       topk_ids=topk_ids,
       topk_weights=topk_weights,
   )
   output = layer(tensors)
   layer.destroy()

The split layer creates its transport lazily on the first forward call. The
process group and CUDA device must be initialized before that call. Call
``destroy()`` collectively on all EP ranks during shutdown.

Split mode
----------

Use :class:`SplitConfig` to compose a communication backend with a local
expert kernel. The split forward path is:

``hidden_states -> dispatch -> local kernel -> combine -> output``

``nccl_ep`` is the default transport when the NCCL-EP extension is built.
``nixl_ep`` is available when the NIXL-EP extension is built and configured.
The transport-specific constraints, quantization knobs, and multi-rank test
commands are documented in the
`MoE-EP runbook <https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/moe_ep_runbook.md>`_.

.. autoclass:: SplitConfig
   :members:

.. autoclass:: MoEEpSplitLayer
   :members:
   :show-inheritance:

Tuning and fault tolerance
--------------------------

Pass fleet-level knobs through ``fleet_knobs`` when constructing a split
layer. They are applied once when the transport fleet is created; the layer
sets the current CUDA stream and routing weights on each forward automatically.

* Use :class:`FleetAlgoKnobNumChannelsPerRank` and
  :class:`FleetAlgoKnobNumQpsPerRank` to tune transport parallelism.
* Use :class:`FleetAlgoKnobRdmaBufferSize` and
  :class:`FleetAlgoKnobTopologyCapacity` for RDMA buffering and a fleet that
  may later change its active rank set.
* Use :class:`FleetAlgoKnobAllocator` to share NCCL-EP buffers with the Torch
  caching allocator when the host framework has a shared memory budget.
* Use :class:`FleetAlgoKnobQuantization` only with a transport/backend that
  documents the selected :class:`QuantType`.
* Use :class:`FleetAlgoKnobFaultTolerance` only with the low-latency split
  algorithm. A masked rank contributes no output and its routing weights are
  not renormalized; applications should account for this when recovering
  from a failed rank.

The runbook gives backend-specific ranges and the fault-recovery state
machine. Treat those recommendations as transport-specific rather than
portable defaults.

.. autoclass:: FleetAlgoKnobNumChannelsPerRank
   :members:

.. autoclass:: FleetAlgoKnobNumQpsPerRank
   :members:

.. autoclass:: FleetAlgoKnobRdmaBufferSize
   :members:

.. autoclass:: FleetAlgoKnobTopologyCapacity
   :members:

.. autoclass:: FleetAlgoKnobAllocator
   :members:

.. autoclass:: FleetAlgoKnobQuantization
   :members:

.. autoclass:: FleetAlgoKnobFaultTolerance
   :members:

Mega mode
---------

Use :class:`MegaConfig` to select a fused expert-parallel kernel. The
``megakernel`` value is a registered architecture-specific configuration. A
mega layer normally preprocesses the canonical weight pack at construction and
allocates its workspace lazily on the first forward call.

For CUDA Graph capture, warm up every EP rank before capture. For workloads
with multiple token capacities, create reusable profiles with
``create_workspace(max_tokens_per_rank)`` before capture. Workspace creation
and destruction are collectives and must be called in the same order on all
ranks.

.. autoclass:: MegaConfig
   :members:

.. autoclass:: MoEEpMegaLayer
   :members:
   :show-inheritance:

.. autoclass:: MoEEpMegaWorkspace
   :members:

Input and weight containers
---------------------------

``hidden_states`` has shape ``[num_tokens, hidden]``. ``topk_ids`` and
``topk_weights`` contain the routing decision for each token. Optional fields
in :class:`MoEEpTensors` carry activation scales and backend-specific
metadata; a backend documents which optional fields it consumes.

The canonical weight pack stores one rank's local experts. Unquantized packs
contain BF16 or FP32 ``w13`` and ``w2`` tensors. Prequantized packs must
provide both scale planes.

.. autoclass:: MoEEpTensors
   :members:

.. autoclass:: MoEWeightPack
   :members:

.. autoclass:: UnquantizedMoEWeights
   :members:

.. autoclass:: PrequantizedMoEWeights
   :members:

.. autofunction:: dummy_moe_weights

Bootstrap and configuration
---------------------------

.. autoclass:: BootstrapConfig
   :members:

.. autoclass:: FleetParams
   :members:

.. autoclass:: EpAlgorithm
   :members:

.. autoclass:: EpLayout
   :members:

.. autoclass:: QuantType
   :members:

Layer factory
-------------

.. autofunction:: MoEEpLayer

Backend availability
--------------------

Use these probes before selecting an optional transport. They return false
when the corresponding extension is not installed or cannot be loaded.

.. autofunction:: available_backends

.. autofunction:: have_nccl_ep

.. autofunction:: have_nixl_ep

.. autofunction:: supports_fault_tolerance

Lifecycle helpers
-----------------

These helpers are useful when a host framework owns process-group and runtime
lifecycle. Automatic bootstrap is enabled by default; set
``BootstrapConfig(auto_bootstrap=False)`` when the host will call the runtime
helpers explicitly.

.. autofunction:: bootstrap_moe_ep_runtime

.. autofunction:: finalize_moe_ep_runtime

.. autofunction:: ensure_moe_ep_cuda_device

Further reading
---------------

* `MoE-EP architecture <https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/moe_ep_architecture.md>`_
* `MoE-EP runbook <https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/moe_ep_runbook.md>`_
* `MoE-EP implementation notes <https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/MoE_EP_impl.md>`_
