PCIe IPC AllGather and ReduceScatter schedules
==============================================

The public entry points are the workspaces, launch configurations, variant
enums, and policy getters exported from ``flashinfer.comm``. Modules whose names
start with ``_`` contain package-internal implementation details. The prefix is a
Python API convention, not a different import mechanism. Applications should
use the public exports; internal workspace, topology, JIT, and tuning helpers
can change without becoming additional public APIs.

Python package layout
---------------------

``flashinfer.comm.pcie_ipc_collectives`` groups the AG/RS Python implementation
and shared PCIe collective helpers:

- ``all_gather.py`` and ``reduce_scatter.py`` provide the workspace APIs;
  ``all_gather_policy.py`` and ``reduce_scatter_policy.py`` define their launch
  configurations, variants, and seed policies.
- ``_ag_rs_workspace.py``, ``_ag_rs_tuning.py``, ``_ag_rs_module.py``, and
  ``_ag_rs_topology.py`` contain AG/RS workspace management, exact-shape tuning,
  JIT loading, and operator-specific topology admission.
- ``_topology.py``, ``_lifecycle.py``, and ``_constants.py`` provide shared
  GPU identity/NVML evidence, stream and resource lifecycle, and constants.
  The existing AllReduce modules reuse these helpers but retain their current
  locations in ``flashinfer.comm`` and their own policy and admission rules.

Applications continue to import the public types from ``flashinfer.comm``.
The general-purpose ``cuda_ipc.py`` utilities remain outside this package.
CUDA headers, TVM-FFI bindings, JIT generators, and tests retain FlashInfer's
existing ``include/``, ``csrc/``, ``flashinfer/jit/``, and ``tests/`` layout.

What a variant describes
------------------------

The existing enum names identify complete implementations. They combine three
choices: the communication algorithm, the transport (SM instructions or CUDA
copy engines), and the amount of work between synchronizations. These are
different dimensions; in particular, ``COPY_ENGINE`` is a recursive-doubling
implementation, and the two ReduceScatter ``ONE_PACK`` implementations also
visit peer destinations in cyclic order. Enum names and values remain stable.

.. list-table::
   :header-rows: 1

   * - Collective / variant
     - Admitted world sizes
     - Schedule and intended tradeoff
   * - AG ``FLAT_PUSH``
     - 4
     - Load a local pack once, push it directly to each peer, synchronize once, then materialize peer shards from local staging. Low synchronization overhead is useful for small shards.
   * - AG ``RECURSIVE_DOUBLING``
     - 2, 4, 8
     - Exchange accumulated shards with one XOR peer per pass, using SM load/store. More pass barriers, with less traffic across the slow boundary on ordered 4+4 TP8.
   * - AG ``COPY_ENGINE``
     - Ordered 4+4 TP8
     - The same XOR ``4, 2, 1`` exchange order using contiguous asynchronous CUDA copies. Bit-reversed staging makes each accumulated region contiguous. Copy submission and synchronization add fixed overhead; larger payloads can benefit from the copy-engine transport.
   * - RS ``FLAT_CYCLIC``
     - 2, 4
     - Send every destination shard directly to its owner in cyclic destination order, synchronize once, then reduce the local staging contributions.
   * - RS ``FLAT_ONE_PACK``
     - 2, 4
     - Use flat owner-push, but process one pack per thread in each grid-sized batch. Keep the local contribution in FP32 registers across publication, then reduce and store that batch.
   * - RS ``TOPOLOGY_CYCLIC``
     - Ordered 4+4 TP8
     - Reduce paired owner shards inside each four-GPU island, exchange one partial with the opposite-island owner, then finish the reduction. One island barrier and one owner-pair barrier follow the two sending phases.
   * - RS ``TOPOLOGY_ONE_PACK``
     - Ordered 4+4 TP8
     - The same hierarchical algorithm in grid-sized batches, retaining the local island partial in FP32 registers across the cross-island exchange. Two barriers are required per batch.

``CYCLIC`` describes destination traversal, not a ring forwarding algorithm.
``ONE_PACK`` means one live pack **per thread**, not one pack for the entire GPU.
A pack is 16 bytes: eight BF16/FP16 values or four FP32 values. With ``B`` blocks
and ``T`` threads per block, each batch covers up to ``16 * B * T`` shard bytes.
For a shard of ``S`` bytes, the flat one-pack kernel has
``ceil(S / (16 * B * T))`` publication barriers per CTA, while the topology
one-pack kernel has twice that many rendezvous per CTA. These are per-CTA
protocol counts; they are not a single GPU-wide barrier count.

The extra barriers explain a possible large-message cost of batching, but do
not determine a universal crossover. The two schedules also differ in register
lifetime, instruction order, memory access order, and the launch geometry that
the tuner selects.

Why use two islands at TP8?
---------------------------

Let ``S`` be one rank-local shard in bytes: an AG input or an RS output. Count
directed peer payload bytes across all eight ranks, excluding local copies,
flags, protocol overhead, and retransmissions.

In a flat schedule every rank sends four shards across the island boundary:
``8 * 4S = 32S`` total cross-island payload. With the hierarchical schedules each
rank sends only one shard or partial across that boundary:
``8 * S = 8S``. Total peer payload remains ``7S`` per rank; communication shifts
from the cross-island links to the island-local links.

- AG starts by exchanging one input shard with XOR peer ``4``. The later XOR
  ``2`` and ``1`` passes relay accumulated shards within each island.
- RS first combines four contributions in each island, then each owner
  exchanges one shard-sized partial with XOR peer ``4``.

This is a traffic model, not a prediction of a fourfold latency improvement.
Latency also depends on barriers, available links, contention, local memory
traffic, launch geometry, and transport. On RS, each transmitted partial uses
the output dtype. BF16/FP16 therefore introduce an intermediate rounding
boundary before the final FP32 accumulation; results need not be bitwise
equal to NCCL or to the flat reduction order.

Topology admission versus mathematical correctness
--------------------------------------------------

The current optimized TP8 admission contract requires logical ranks ``0..3``
and ``4..7`` to be the two islands. UUID-based NVML evidence must show unique
physical GPUs on one host, non-SYSTEM links within each island, and SYSTEM
links across islands. Both directions of every pair must agree. Missing or
conflicting evidence leaves the optimized paths unavailable.

That fixed order matches the kernels' rank arithmetic and the deployment
whose performance motivated these schedules. SYSTEM is a physical topology
classification, not a condition in the algebra of a sum or a copy. This check
does not prove that an admitted configuration is fastest, and it does not
replace CUDA IPC setup. Other placements would require a separately supported
schedule or a rank mapping that preserves the caller's rank-major tensor
layout. The present API does not automatically reorder ranks.

AG recursive doubling remains available when this proof fails. TP8 RS reports
unsupported; callers can select their own fallback. At TP2/TP4 the flat RS
paths do not require the 4+4 proof. ``supports()`` describes this implementation's
admission contract, not a performance comparison with NCCL.

Defaults and measured selection
-------------------------------

Without a matching tuning entry, the policy returns a deterministic seed:

.. list-table::
   :header-rows: 1

   * - RS world size
     - Seed variant by rank-local output bytes
   * - 2
     - ``FLAT_ONE_PACK`` below 4 MiB; ``FLAT_CYCLIC`` at or above 4 MiB
   * - 4
     - ``FLAT_ONE_PACK``
   * - 8, ordered 4+4
     - ``TOPOLOGY_ONE_PACK`` below 128 KiB; ``TOPOLOGY_CYCLIC`` at or above 128 KiB

These boundaries are heuristic seeds. The historical RTX 6000D H6144/BF16
grid selected TP2 one-pack at 3 MiB and cyclic at 6 MiB, and TP8 one-pack at
96 KiB and cyclic at 192 KiB. The current defaults lie within those sampled
intervals, but no exact-boundary calibration or derivation is recorded. Those
observations do not establish the best variant at exactly 4 MiB or 128 KiB,
for another dtype, or on another PCIe fabric.

The later BF16 campaign at commit
``80f5ef1a5116ba777d40599724c38ae6610a8659`` exercised H4096/H6144 and powers-of-two
row counts from 1 through 32768 on RTX 6000D. Every variant above was selected
for at least one shape. Some selected RS variants disagreed with the defaults, including
topology one-pack at TP8/H4096/16 rows (128 KiB). A selected implementation
beating NCCL does not by itself establish its margin over another custom
variant. A robust deletion or threshold change needs repeated direct
comparisons of the affected variants with their own useful launch geometries.

``tune()`` enumerates every variant legal for the actual workspace and shape,
including the fixed copy-engine configuration, and checks correctness before
timing. It searches useful block/thread combinations within workspace limits;
the seed is included. The tuner measures eager launches and minimizes the
maximum elapsed time across ranks. Its exact-shape cache also records dtype,
world size, placement fingerprint, and workspace limits. This is not an
exhaustive graph-mode performance oracle. Tuning can choose a variant on
either side of the seed thresholds.

Validation coverage
-------------------

The GPU tests explicitly execute each admitted variant for BF16, FP16, and
FP32, including a partial final pack batch. RS additionally compares every
variant against an FP32 reference with dtype-appropriate tolerances.

Each core test also tunes one small BF16 shape, checks the default call against
the reference, and repeats the check with a fresh workspace using the saved
cache. This is a functional smoke test, not a performance assertion.

The graph tests queue mixed variants, changing grids and shapes, and
then replay captured sequences with changed inputs and independent output
snapshots. On TP8, the live topology gate controls whether CE/topology variants
are exercised. Set ``FLASHINFER_TEST_PCIE_IPC_ORDERED_4PLUS4=1`` on a designated
4+4 test node to require that coverage; arbitrary CI machines are not assumed
to have that placement. Performance conclusions require separate measurements.
