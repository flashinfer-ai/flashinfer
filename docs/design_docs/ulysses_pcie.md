# Experimental PCIe Ulysses backend

`UlyssesCommunicator(backend="pcie")` is an explicit, single-node transport for PCIe-connected GPU groups; `backend="auto"` never selects it. One rank is an identity path; two ranks use CUDA peer copies; four or eight ranks prefer an all-RDMA route (every peer's payload over the rank-local mlx5 RC QP with an interleaved UMR) and fall back to all-P2P when the mlx5/GPUDirect requirements are not met; `FLASHINFER_ULYSSES_PCIE_ROUTE` forces the all-P2P route, the all-RDMA route at any multi-rank world size, or the eight-rank 4+4 NUMA hybrid (same-NUMA CUDA peer copies, cross-NUMA mlx5). Every route lands the Ulysses transforms directly in their final layouts, with no pack/unpack or staging output tensor.

Why copy engines rather than SM stores: the fused-transpose kernel behind `backend="nvlink"` (`include/flashinfer/comm/ulysses_all_to_all.cuh`) lets SM threads write into a peer-visible staging buffer, which is right for NVLink but degenerates into small transactions across a PCIe host bridge — 5-13x slower than this backend's all-P2P copy-engine route on the 8-GPU PCIe node it targets, at equal (bit-identical) outputs. (Measured 2026-08 during bring-up on the reference node — 8x RTX PRO 5000, dual-NUMA 4+4, PCIe Gen5 — with a one-off engine-comparison script that was not kept; only this conclusion survives.)

## Support matrix

| Property | PCIe backend |
|---|---|
| Hosts / ranks | One host; world size 1, 2, 4, or 8 |
| Routes | ws 1: identity (no JIT, no transport); ws 2: all-pairs CUDA P2P; ws 4/8: all-RDMA preferred, all-P2P fallback; `FLASHINFER_ULYSSES_PCIE_ROUTE` forces p2p or rdma at ws 2/4/8, or the 4+4 NUMA hybrid at ws 8; full-group CUDA P2P required on every route |
| RDMA route requirements | One mlx5 device per rank with DEVX, RC QP, UMR, an active IPv4 RoCE v2 GID, GPUDirect RDMA; batch size 1 and head-row pitch `H * D * element_size <= 65,535` bytes; hybrid additionally needs the 4+4 NUMA split |
| Build requirements | rdma-core >= 36 (libibverbs and libmlx5 headers/libraries) and the CUDA driver library, on every route including all-P2P |
| Input | Contiguous 4-D CUDA tensor with a 1-, 2- or 4-byte element type (FP16/BF16/FP32, FP8, INT8/UINT8); any batch size on the all-P2P route |
| Execution | P2P enqueues asynchronously on the caller stream; the RDMA routes block the host until every rank reaches the barrier (unbounded) and until RDMA completes (fixed 10 s deadline); one in-flight operation per communicator, bound to the stream of the first call |
| CUDA Graphs | All-P2P: capturable when every output comes from `allocate_output`; hybrid/all-RDMA: refused |

Full-group CUDA P2P is required even on the RDMA routes because every exchange's opening and closing epoch barriers read CUDA-mapped signals from all ranks. All routes share one translation unit, so the JIT compile/link needs the libibverbs and libmlx5 headers/libraries and the CUDA driver library (`-lcuda`) even when topology selects all-P2P; runtime verbs requirements apply only to the RDMA routes. The transport calls `ibv_query_gid_ex` (rdma-core 35.0), `ibv_reg_dmabuf_mr` and the interleaved-MKey `mlx5dv_wr_mkey_configure` / `mlx5dv_wr_set_mkey_layout_interleaved` family (rdma-core 33.0), so build against rdma-core >= 36; the reference node runs 39.0. `missing_ulysses_pcie_dependencies()` checks only that the libraries are present, so an older rdma-core surfaces as a JIT compile or link failure naming the missing symbol rather than as a route-planning fallback. Topology probing selects one GID index per rank and native initialization revalidates it (port active, RoCE v2, IPv4-mapped) before creating QPs. A forced backend raises collectively when no route is available.

## Device and lifecycle

```python
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
dist.init_process_group("nccl", device_id=device)

with UlyssesCommunicator(
    max_elems=q.numel(), dtype=q.dtype, backend="pcie", device=device
) as comm:
    q_out = comm.allocate_output(q, op="scatter_heads")
    comm.scatter_heads(q, out=q_out)
```

Multi-rank construction, each new output registration, and `close()` are collective, and all ranks must issue public calls in the same order. Teardown first verifies on every rank that native work is bounded, then closes peer imports group-wide, and only then releases local registrations and the transport — so no rank frees an export a peer may still reference. A failed `close()` raises on all ranks and may be retried.

On the RDMA routes (hybrid and all-RDMA) the barrier rendezvous is unbounded — a slow rank (a cold JIT compile, a long host-side stall) only delays its peers — while a rank that fails while running publishes a sticky abort to every peer and drains posted work requests; the communicator is then broken group-wide and must be closed and rebuilt. RDMA completion is bounded by a fixed 10 s deadline, so a stalled NIC breaks the group the same way instead of hanging it. If the abort drain cannot be confirmed, `close()` refuses to synchronize or unmap and the process must exit (see `csrc/ulysses_pcie_transport.cuh` for the protocol). On either route a rank that exits or diverges from the group call order is not detectable and leaves its peers waiting; the launcher must terminate the job.

## Output lifetime

Multi-rank PCIe calls require an explicit output from `allocate_output`; allocate one per live result and geometry during setup:

```python
q_out = comm.allocate_output(q, op="scatter_heads")
k_out = comm.allocate_output(k, op="scatter_heads")
v_out = comm.allocate_output(v, op="scatter_heads")

comm.scatter_heads(q, out=q_out)
comm.scatter_heads(k, out=k_out)
comm.scatter_heads(v, out=v_out)
attn_out = attention(q_out, k_out, v_out)
output = comm.allocate_output(attn_out, op="gather_heads")
comm.gather_heads(attn_out, out=output)
```

An output is overwritten by its next operation; use a different output for every result that must coexist. Outputs stay registered until `close()` — dropping the Python view frees nothing — and must not be passed to another communicator or overlap the input.

On the RDMA routes each output also owns a landing buffer of the same capacity, and every exchange copies the operand into it before the NIC reads it: registering caller memory instead would require `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS` on the whole backing allocation (for a PyTorch tensor, a caching-allocator segment shared with unrelated tensors) for the lifetime of the MR. The all-P2P route reads the operand in place.

## CUDA Graphs

The all-P2P route replays correctly because the barrier advances its epoch in device memory (`AdvanceEpoch`), not as a launch argument: a baked-in epoch would make every replay fall through both barriers silently. (`pcie_ipc_all_reduce.cuh` advances its scratch epoch the same way.) `allocate_output` refuses capture because it is collective — allocate before capturing. The RDMA routes refuse capture outright: they post and poll mlx5 work requests from the host. `FLASHINFER_ULYSSES_PCIE_ROUTE=p2p` on every rank gives a capturable all-P2P route.

## Routing controls

- `FLASHINFER_ULYSSES_PCIE_ROUTE`: `auto` (default: world sizes 4 and 8 prefer all-RDMA — from 4 ranks up, all-P2P crosses host bridges and measured well below the RDMA route on the reference node, while 2 ranks share a PCIe switch and P2P wins), `p2p` (force all-P2P), `rdma` (force all-RDMA at any multi-rank world size), or `hybrid` (force the eight-rank 4+4 NUMA hybrid). A forced RDMA route falls back to all-P2P with a `RuntimeWarning` when its requirements are not met.
- `FLASHINFER_ULYSSES_PCIE_NICS`: comma-separated mlx5 device names, one per rank in rank order; overrides automatic PCI-distance NIC routing.
- `FLASHINFER_ULYSSES_PCIE_GID_INDICES`: comma-separated GID table indices, one per rank in rank order; chooses among usable IPv4 RoCE v2 entries when a NIC has several.

Every rank must set them identically.

## Benchmarking

```bash
torchrun --standalone --nproc-per-node=8 \
  benchmarks/comm/bench_ulysses_pcie.py --seq-len 37888 --num-heads 56 --head-dim 128
```

The script validates one scatter_heads and one gather_heads result element-wise against the torch all_to_all_single reference before timing, then times whole loops with the host clock, max-reduces across ranks, and reports the median and spread over trials (see `timed()` for why per-call CUDA events are not usable here). Collective time is not an end-to-end model gain: validate the layout inside the real model before drawing deployment conclusions.
