# TODO: multi-node support for the mega_moe path (cutedsl backends)

Status: analysis only, nothing implemented yet (2026-07-14).

The mega path currently assumes all EP ranks live on one node (or at least
one NVLink/p2p domain). The blockers split into two tiers: host-side
rank/device bookkeeping (cheap to fix) and the kernel's comm mechanism
itself (the real work).

## Tier 1 — host-side rank computation / bootstrap

1. **CUDA device binding falls back to the global rank** —
   `core/runtime/bootstrap.py` (`_ensure_cuda_device`, also repeated in
   `_ensure_torch_dist` and `_init_nvshmem_after_dist`):

   ```python
   local_rank = int(os.environ.get("LOCAL_RANK", str(bootstrap.rank)))
   torch.cuda.set_device(local_rank)
   ```

   Without `LOCAL_RANK` exported, rank 8 on node 1 tries `cuda:8` and
   crashes. The kernel-side bootstrap (`kernel_src/.../src/src/bootstrap.py`
   `_discover_ranks`) already handles this properly — it recognises both
   torchrun (`RANK`/`LOCAL_RANK`) and SLURM (`SLURM_PROCID`/`SLURM_LOCALID`,
   mirrors `SLURM_NTASKS_PER_NODE` → `LOCAL_WORLD_SIZE`) — but the moe_ep
   layer path (`bootstrap_moe_ep_runtime`) only reads `LOCAL_RANK` with the
   global-rank fallback. Fix: derive local rank the way `_discover_ranks`
   does (or add `local_rank` to `BootstrapConfig` so embedding frameworks
   can pass it explicitly), and never default to the global rank.

2. **NVSHMEM UID broadcast `src` is wrong for EP subgroups** —
   `_init_nvshmem_after_dist` does
   `uid = nvshmem.core.get_unique_id(empty=(rank != 0))` with the
   *EP-group* rank, but then `dist.broadcast(uid_tensor, src=0, group=pg)` —
   torch's `src` is a *global* rank. Works today only because the EP group
   is WORLD (or its rank-0 happens to be global 0). Fix:
   `src=dist.get_global_rank(pg, 0)`. Not strictly a multi-node bug, but it
   bites exactly when hosts start passing non-WORLD process groups
   (multi-node PP×EP layouts).

3. **Single-node-only dist fallback is fine as-is** — `_ensure_torch_dist`
   only auto-inits gloo at `world_size == 1`; multi-node requires the caller
   to init torch.distributed first, which is already the documented
   contract.

## Tier 2 — the kernel comm mechanism (the real blocker)

The dispatch/combine transport is **direct NVLink load/store to peer-mapped
symmetric-heap addresses**, not NVSHMEM RDMA:

- `_compute_peer_offsets` (`shim/comm.py`) calls
  `nvshmem.core.get_peer_tensor(sym_tensor, peer)` for **every** peer in
  `world_size` and reduces each to one constant pointer delta.
- The kernel maps local→peer addresses with that delta
  (`src/src/sym_buffer.py` `SymBuffer.map`, `num_max_ranks` is a compile-time
  constexpr) and communicates via raw PTX ld/st and sys-scope atomics
  (`red.add.release.sys`, `cp.reduce.async.bulk`) — see
  `src/src/token_comm.py`, plus the NVLink barrier
  (`nvlink_barrier_counter`).

`get_peer_tensor` only yields a usable pointer for PEs reachable via CUDA
p2p mapping. Across IB-connected nodes there is no mapped peer pointer and
no IBGDA/proxy fallback in the kernel — so classic multi-node (IB/RoCE) is
**not** a config fix; it needs a new transport path in the kernel
(NVSHMEM device-API puts/gets or a proxy-based dispatch), which is a kernel-
team-scale work item.

The realistic near-term target is **multi-node NVLink (MNNVL / GB200
NVL72)**, where the p2p-mapped scheme can extend across nodes unchanged in
principle. To validate that:

1. Confirm `nvshmem.core.get_peer_tensor` returns mapped pointers for
   cross-node PEs on an MNNVL fabric (NVSHMEM must be built/configured with
   MNNVL support; check `NVSHMEM_DISABLE_P2P` / fabric-detection behavior).
2. Confirm the constant-delta assumption ("one offset per peer valid for
   every sub-region") still holds for fabric-mapped addresses — it relies on
   all ranks performing identical collective allocations in the same order.
3. Audit sys-scope atomics + `fence`/barrier phases over the fabric
   (`red.add.*.sys` is defined for p2p-mapped memory incl. MNNVL, but the
   barrier/flag protocol's performance assumptions are intra-node-tuned;
   `flag_batch` / `epi_flag_batch` knobs and the autotune candidate set will
   likely need fabric-specific profiles).
4. `num_max_ranks` is a constexpr with a `<= 16` by-val fast path
   (`sym_buffer.py` `_BYVAL_RANK_LIMIT`); world sizes beyond 16 switch IR
   representation and are untested — NVL72-scale EP (e.g. 32–72 ranks)
   needs a pass over that path and the workspace sizing
   (`world_size * num_tokens_per_rank` receive pools).

## Suggested order of work

1. Tier-1 fixes (local-rank derivation, subgroup UID broadcast src) — small,
   land independently; they also unblock multi-node *testing* of everything
   else.
2. Add multi-node validation guard: until Tier 2 is proven, detect
   cross-node EP groups at init (`LOCAL_WORLD_SIZE < world_size` without an
   MNNVL fabric) and fail with a clear "mega path is single-NVLink-domain
   only" error instead of a null peer-pointer crash.
3. MNNVL bring-up per the checklist above (needs GB200/NVL72 time).
4. IB multi-node (new kernel transport) — track as a separate project with
   the kernel team; out of scope for the shim.

## Related

- CUDA-graph support (landed: warmup contract + capture guards; see
  `tests/moe_ep/test_mega_cuda_graph*.py`); interacts here: symmetric
  heap allocation and peer-pointer computation are host collectives that
  must stay pre-capture regardless of node count, and multi-node replay
  needs all ranks (across nodes) replaying in lockstep.
