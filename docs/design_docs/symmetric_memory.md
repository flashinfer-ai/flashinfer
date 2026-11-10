# Symmetric Memory over NVLink

**Scope**: `MnnvlMemory` and `SymmDeviceMemory` in `flashinfer/comm/mnnvl.py`:
how they allocate and share memory, their lifecycle and checkpoint/restore, and
how the NVLink one-sided all-to-all gates CFT on the handle type.

FlashInfer's NVLink collectives work on symmetric memory: every rank of a group
allocates a buffer of the same size, and every rank maps every peer's buffer
into its own virtual address space, so a kernel reads and writes peer memory
with plain loads and stores.

A **group** is the set of ranks that share one allocation: the ranks of the
`CommBackend` it is allocated over. Allocations are collective over their
group, and a process can be in several groups at once, for example a TP group
for all-reduce and an EP group for MoE all-to-all, each with its own
allocations.

<p align="center"><img src="./media/symmetric_memory.png" alt="Symmetric memory across GPUs" width="600"/></p>
<p align="center"><em>Symmetric memory registration across GPUs. (Image source: <a href="https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/">NVIDIA NCCL 2.27 blog</a>)</em></p>

Two classes in [`flashinfer/comm/mnnvl.py`](../../flashinfer/comm/mnnvl.py)
provide it, `MnnvlMemory` and `SymmDeviceMemory`. They allocate and share the
physical memory the same way and differ in how they lay it out; the
[comparison](#comparison) at the end summarizes the differences.

## Allocating and sharing the physical memory

Each rank backs its buffer with `cuMemCreate` and exports a shareable handle;
the peers import it and map it. The handle type decides how the handles
travel:

- **FABRIC** handles are opaque bytes that any rank in the NVLink domain can
  import, on any host. They move through the `CommBackend`'s object
  collectives. They need an IMEX channel and, across hosts, the `nvidia-imex`
  service on every node.
- **POSIX file descriptors** move between processes over AF_UNIX sockets with
  `SCM_RIGHTS` ([`fd_exchange.py`](../../flashinfer/comm/fd_exchange.py)), so
  they work only within one host. They need no IMEX channel.

`allocate_mnnvl_handles` does this for both classes. It is collective over the
group, and every rank of the group always takes the same path:

```text
allocate_mnnvl_handles(comm, dev_id, make_prop)
    handle_type, single_host = choose_mnnvl_handle_type(comm, dev_id)
    attempt(handle_type)
    if it failed, handle_type is FABRIC and single_host:
        attempt(POSIX)
    if it still failed: raise with every rank's error

choose_mnnvl_handle_type(comm, dev_id)      # one allgather of (host, reason, cluster UUID)
    one host:   FABRIC if every GPU reports FABRIC support, else POSIX
    many hosts: FABRIC if every GPU reports FABRIC support and the same
                non-zero NVML fabric cluster UUID, else raise with each rank's reason

attempt(handle_type)                        # _create_and_import_handles
    create and export this rank's allocation
    vote; if any rank failed: release, return the errors
    exchange the handles; import every peer's
    vote; if any rank failed: release everything, return the errors
```

Every rank of the group must pick the same handle type, since the two types
exchange handles through different collectives. Each call therefore decides
afresh from one allgather over the group, rather than reusing an earlier
result: a process that already allocated in another group (say, its TP group)
may have settled on a type that the other ranks of this group (say, its EP
group) never saw. The votes likewise turn a failure on one rank into the same
outcome on all of them instead of leaving the others blocked in a collective.

Each caller passes a callback, `make_prop(handle_type)`, that returns the
allocation properties and size for a handle type; `allocate_mnnvl_handles`
calls it for each type it attempts, since the handle type is part of the
properties. `SymmDeviceMemory` sizes its buffer there, including the multicast
granularity. `MnnvlMemory` must place the allocation in its address reservation
before the handle type is settled, so it sizes allocations with a granularity
that fits both handle types.

## `MnnvlMemory`

```python
MnnvlMemory.initialize()
MnnvlMemory.set_comm_from_config(mapping, MnnvlConfig(comm_backend=backend))  # optional
memory = MnnvlMemory(mapping, size)
workspace = memory.as_torch_strided_tensor(torch.uint8)  # [group_size, size], row i = rank i
```

The group comes from the `Mapping`, not from the constructor: the first
allocation splits the communicator into the ranks with the same PP and CP rank,
ordered by TP rank, i.e. the TP group, which contains the MoE EP group. The
communicator is MPI unless `set_comm_from_config` supplies another
`CommBackend`, and it is cached on the class, so every later allocation of the
process uses the same group. `size` is the bytes per rank and must be equal on
all ranks; the memory lives on the current CUDA device.

`open_mnnvl_memory` checks that every rank asks for the same size, rounds it
up to the granularity, reserves a new address range when the current one is
full, and maps every rank's allocation at `start + i * rank_stride + offset`.
`allocated_map` keeps a record per allocation (handles, offsets, handle type)
and `address_refcnt` counts the allocations of each reservation.
`fabric_handles` tells whether an allocation uses FABRIC handles.

## `SymmDeviceMemory`

```python
memory = SymmDeviceMemory(
    buf_size,                 # bytes per rank, equal on all ranks
    group_size, group_rank,   # this rank's place in the group
    device_idx,               # the CUDA device to allocate on
    comm_backend,             # CommBackend of the group; MPI if None
    enable_multicast=True,
    allocate_signal_pads=True,
)
memory.get_unicast_ptr(i)     # rank i's buffer
memory.get_multicast_ptr()    # stores reach every rank's buffer
```

Each instance names its group explicitly: `comm_backend` is the group, and
`group_size` and `group_rank` must match its size and this rank's rank in it.
Instances over different backends are independent.

After `allocate_mnnvl_handles`, the unicast buffers are mapped back to back in
the instance's reservation. With multicast enabled, rank 0 creates the
multicast object with the same handle type as the unicast buffers and
broadcasts its handle; every rank imports it, maps it and binds its own buffer
to it. Each buffer is followed by a signal pad, zeroed by its owner, that the
all-reduce kernels use for cross-rank flags.

## Lifecycle

```text
construct (collective) ──► mapped ──► release physical memory (collective) ──► unmapped
                             ▲                                                    │
                             └─────── restore over a new CommBackend (collective) ─┘

mapped or unmapped ──► last reference dropped (local) ──► freed
```

**Creation** is collective over the group: every rank constructs the object
with the same size, in the same order relative to other collectives on that
group. The memory starts uninitialized. Each rank initializes its own buffer
(for example the all-to-all metadata, or `lamport_initialize`), and the group
must synchronize before any peer writes into it.

**Destruction** happens in `__del__`, when the last Python reference goes away.
It is local: no collective and no device synchronization.

- `MnnvlMemory` unmaps every rank's segment and releases the handles. The
  address reservation is freed with its last allocation.
- `SymmDeviceMemory` additionally unmaps and releases the multicast object and
  frees its pointer arrays and reservations.

CUDA frees a rank's physical memory only once every rank that imported it has
unmapped and released it, so a peer that tears down early does not invalidate
this rank's mappings. The owner must still make sure that no kernel uses the
memory anymore, on this rank or on its peers. During interpreter shutdown both
destructors skip the cleanup and leave it to process exit.

**Owners** decide how long the memory lives:

| Owner | Memory | Lifetime |
|---|---|---|
| `NVLinkOneSidedAlltoAll`, `CakeAlltoAll` | `MnnvlMemory` workspace shared by the instances of one EP group and layout, refcounted | Freed by `destroy()` of the last instance, after a device synchronization; every rank must have finished its last combine on it |
| `NVLinkTwoSidedAlltoAll` | Two `MnnvlMemory` workspaces per EP rank and size | Kept for the process and reused by later instances |
| All-reduce workspaces (`trtllm_ar`, `trtllm_mnnvl_ar`) | `SymmDeviceMemory` | Dropped by the workspace's `destroy()` |

**Checkpoint/restore** releases the physical memory but keeps the virtual
addresses, so CUDA graphs captured on them stay valid. Both steps are
collective, and the restored memory is uninitialized.

- `MnnvlMemory`: `unmap_mnnvl_memory` synchronizes the device, waits for every
  rank, releases the memory and waits again; `remap_mnnvl_memory` allocates
  over a new `CommBackend` with the same rank and size, which replaces the old
  one. Both are in
  [`nvlink_common.py`](../../flashinfer/moe_ep/backends/split/comm/nvlink_common.py).
- `SymmDeviceMemory`: `_unmap_and_release_handles` synchronizes the device,
  waits for every rank, then unbinds and releases the multicast object and the
  unicast buffers; `_create_and_map_handles` allocates them again and rebuilds
  the multicast object at the same addresses.

The restore runs `allocate_mnnvl_handles` again, so it can settle on a
different handle type than the original allocation. The workspaces expose
these steps as `checkpoint_prepare` and `checkpoint_restore` (see
[`docs/api/comm.rst`](../api/comm.rst)).

## Interaction with CFT counted writes

The NVLink one-sided all-to-all can bind its workspace to CUDA logical
endpoints (CFT counted writes), which only accept memory shared through
FABRIC handles. The workspace layout depends on whether CFT is used, so the
backend decides before allocating, from `MnnvlMemory.uses_fabric_handles` (the
precheck above). If the allocation then falls back to POSIX file
descriptors, binding is skipped and the backend uses its fence path.

## Comparison

| | `MnnvlMemory` | `SymmDeviceMemory` |
|---|---|---|
| Users | MoE all-to-all workspaces (`flashinfer.moe_ep` NVLink one-sided, Cake, two-sided), `dcp_alltoall` | All-reduce workspaces (`trtllm_ar`, `McastGPUBuffer` for `trtllm_mnnvl_ar`), the SM90 push-style MegaMoE |
| Group | Derived from a `Mapping` and cached on the class (`MnnvlMemory.get_comm`); one group and one device per process | The `CommBackend` passed to the constructor (MPI by default) |
| Virtual addresses | One reservation per process, shared by allocations: `group_size * rank_stride` bytes, the stride a multiple of 512 MB; allocations are packed at increasing offsets and the reservation is freed when its last allocation closes | One reservation per instance: `group_size * allocation_size` bytes, rank `i` at offset `i * allocation_size` |
| Peer `i` of an allocation at `ptr` | `ptr + i * rank_stride` | `get_unicast_ptr(i)` |
| Multicast | No | Optional: one multicast object bound to every rank's buffer, mapped at `get_multicast_ptr()` |
| Extras | — | A signal pad after each rank's buffer; Lamport initialization for the one-shot all-reduce |
| Freed | Local `__del__`; the reservation with its last allocation | Local `__del__`, together with its reservations and multicast object |
| Checkpoint/restore | `unmap_mnnvl_memory` / `remap_mnnvl_memory` | `_unmap_and_release_handles` / `_create_and_map_handles` |
