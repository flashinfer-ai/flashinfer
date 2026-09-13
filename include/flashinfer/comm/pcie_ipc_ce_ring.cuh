/*!
 * Copy-engine ring all-reduce: a second data plane beside the SM kernels.
 *
 * The seven kernels in pcie_ipc_all_reduce.cuh move every byte with SM
 * load/store and synchronise through the payload itself: +0.0 means "not yet
 * written", so a producer rewrites any real +0.0 to -0.0 and a consumer polls
 * its own 16-byte pack. A copy engine can do neither. It cannot transform data,
 * so that rewrite would need an SM pass over the whole payload -- most of what
 * the engine was meant to save -- and it reports only bulk completion, so it
 * cannot observe a per-pack sentinel either. Readiness therefore leaves the
 * payload: copy on a side stream, event, a single-thread kernel publishes a
 * monotonic flag to the peer, another spins on it. This plane shares the
 * workspace and the policy layer with the SM kernels and nothing else; those
 * seven are untouched and remain the decode fast path.
 *
 * Which plane wins is a per-fabric, per-shape question the tuner settles bucket
 * by bucket; neither is the default. The argument for this one is scaling, not
 * raw speed: the SM plane's margin over NCCL narrows as the rank count grows on
 * some fabrics, where the ring keeps its own.
 *
 * Two ranks are excluded -- that schedule is one reduce-scatter hop and one
 * all-gather hop with nothing to pipeline against, so scheduling jitter lands
 * directly on the wall clock, and where it was measured the run-to-run spread
 * came out wider than the ring's whole margin. The tuner measures each
 * candidate once per round and cannot act on an advantage smaller than the
 * candidate's own variance. The two-hop structure holds on any fabric; the
 * size of the margin it has to clear was only established on one.
 */
#ifndef FLASHINFER_COMM_PCIE_IPC_CE_RING_CUH_
#define FLASHINFER_COMM_PCIE_IPC_CE_RING_CUH_

#include "pcie_ipc_all_reduce.cuh"

// This file follows the surrounding style: plain cudaError_t returns, no
// framework headers. Scoped to the file and undefined at the bottom.
#define FI_CE_CHECK(expr)                         \
  do {                                            \
    const cudaError_t _fi_ce_e = (expr);          \
    if (_fi_ce_e != cudaSuccess) return _fi_ce_e; \
  } while (0)

namespace flashinfer {
namespace comm {
namespace pcie_ipc {

// Streams and events the ring owns. They live on the handle, not in Python: a
// captured graph keeps references to recorded events, a lifetime rule Python
// cannot enforce. Created in pcie_ipc_init, destroyed in pcie_ipc_dispose.
struct CeResources {
  cudaStream_t copy_stream;
  cudaStream_t flag_stream;
  cudaEvent_t input_ready;
  cudaEvent_t copy_done;
  cudaEvent_t flag_done;
  cudaEvent_t add_done[kCeMaxPieces];
  cudaEvent_t copied[2 * (kMaxWorldSize - 1) * kCeMaxPieces];
};

// Announce to one peer that this rank's copy into its slot has landed.
//
// grid(1) block(1): the value published is this rank's own monotonic count for
// the slot, read and advanced on the device. Nothing is supplied by the host,
// which is what lets a captured graph replay without patching.
//
// Nothing here orders the copy-engine writes -- the stream dependency does.
// This kernel runs on a stream that already waited on the memcpy's event, so
// the payload has landed by the time it is dispatched. The release store only
// has to make the counter store, and everything this device had already
// retired, visible to the peer's acquire.
__global__ void ce_publish_flag_kernel(int32_t* peer_flag, int32_t* send_counter) {
  const int32_t next = load_volatile_i32(send_counter) + 1;
  store_volatile_i32(send_counter, next);
  store_release_i32(peer_flag, next);
}

// The matching wait on this rank's own flag.
//
// generation_reached() is the wrap-safe comparison the barrier kernels already
// use; a plain `observed < expected` is correct for two billion calls and then
// releases early.
__global__ void ce_wait_flag_kernel(int32_t* self_flag, int32_t* wait_counter) {
  const int32_t expected = load_volatile_i32(wait_counter) + 1;
  store_volatile_i32(wait_counter, expected);
  while (!generation_reached(load_acquire_i32(self_flag), expected)) {
  }
}

// out = a + b over whole 16-byte packs.
//
// No sentinel and no barrier: the flag wait that precedes this on the same
// stream is the entire synchronisation. The reduce happens in the payload
// dtype, one rounding per ring hop, which matches NCCL's ring rather than the
// SM kernels' single multi-source reduce -- a different summation order, which
// is already the situation against NCCL and is why the tests compare exact
// small integers.
template <typename T>
__global__ void ce_add_kernel(uint4* out, const uint4* a, const uint4* b, int64_t num_packs) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < num_packs;
       i += stride) {
    out[i] = packed_add_u4<T>(a[i], b[i]);
  }
}

// Sub-chunk depth. Deeper chunking keeps the copy engine busy -- piece c+1's
// copy overlaps piece c's wait and reduce -- but each piece adds a flag round
// trip and two launches, so it stops paying once a piece is small: at the
// large payloads where the depths were compared, going past two was worse on
// both fabrics. Between the shallow depths there is no answer that transfers:
// which one wins flips with the fabric and with the payload size, so two is a
// starting point and nothing more; the tuner settles it per shape.
//
// Must be a pure function of the shape: a rank that picks a different depth
// uses different flag slots and the group hangs with no timeout.
inline int ce_pick_pieces(int64_t shard_elems, size_t shard_bytes, int pack_elems, int requested) {
  const int want = (requested >= 1 && requested <= kCeMaxPieces) ? requested : 2;
  for (int pieces = want; pieces > 1; --pieces) {
    if (shard_elems % (static_cast<int64_t>(pieces) * pack_elems) == 0 &&
        shard_bytes / static_cast<size_t>(pieces) >= (512u << 10)) {
      return pieces;
    }
  }
  return 1;
}

namespace detail {

inline size_t ce_slot_stride(const WorkspaceLayout& layout, int world_size) {
  const size_t shard = layout.max_payload_bytes / static_cast<size_t>(world_size);
  return (shard + 127u) & ~static_cast<size_t>(127u);
}

}  // namespace detail

// Flat neighbour ring: reduce-scatter then all-gather, every step writing only
// to (rank+1)%N so each GPU has exactly one outbound and one inbound stream.
//
// There is no upfront copy of the input into the output. The first send reads
// the caller's buffer directly and every reduce-scatter step is a first touch,
// out[chunk] = in[chunk] + scratch, so the accumulation base folds into the add
// instead of costing a full-size copy on the critical path. Note the `a` operand
// is always the *input*: the scratch already carries the upstream ranks'
// accumulated partial for that chunk, and this rank adds only its own.
template <typename T>
cudaError_t ce_ring_all_reduce_flat(const T* input, T* output, int64_t numel,
                                    const PeerViews& views, int rank, int world_size,
                                    const WorkspaceLayout& layout, const CeResources& ce,
                                    int pieces_hint, int threads, cudaStream_t stream) {
  constexpr int kPackElems = PackTraits<T>::kPackElems;
  const int64_t shard_elems = numel / world_size;
  const size_t shard_bytes = static_cast<size_t>(shard_elems) * sizeof(T);
  const int pieces = ce_pick_pieces(shard_elems, shard_bytes, kPackElems, pieces_hint);
  const int64_t piece_elems = shard_elems / pieces;
  const size_t piece_bytes = static_cast<size_t>(piece_elems) * sizeof(T);
  const size_t slot_stride = detail::ce_slot_stride(layout, world_size);
  const int steps = 2 * (world_size - 1);
  const int next = (rank + 1) % world_size;
  const int prev = (rank - 1 + world_size) % world_size;

  const int64_t packs = piece_elems / kPackElems;
  const unsigned grid =
      static_cast<unsigned>(packs < threads ? 1 : (packs + threads - 1) / threads);
  const unsigned add_grid = grid > 64u ? 64u : grid;

  auto flag_at = [&](int peer, int slot) {
    return reinterpret_cast<int32_t*>(views.ce_flags[peer] +
                                      static_cast<uint64_t>(slot) * kCeFlagStride);
  };
  auto scratch_at = [&](int peer, int k, int p) {
    return reinterpret_cast<char*>(views.ce_scratch[peer]) +
           static_cast<ptrdiff_t>(k) * static_cast<ptrdiff_t>(slot_stride) +
           static_cast<ptrdiff_t>(p) * static_cast<ptrdiff_t>(piece_bytes);
  };

  // Fork: pull both side streams into whatever ordering (or capture) the
  // caller's stream is under. Without this a capture fails outright and an
  // eager call races the caller's producer.
  FI_CE_CHECK(cudaEventRecord(ce.input_ready, stream));
  FI_CE_CHECK(cudaStreamWaitEvent(ce.copy_stream, ce.input_ready));
  FI_CE_CHECK(cudaStreamWaitEvent(ce.flag_stream, ce.input_ready));

  for (int k = 0; k < steps; ++k) {
    const bool reduce_phase = k < world_size - 1;
    int send_c, recv_c;
    if (reduce_phase) {
      send_c = ((rank - k) % world_size + world_size) % world_size;
      recv_c = ((rank - k - 1) % world_size + world_size) % world_size;
    } else {
      const int kk = k - (world_size - 1);
      send_c = ((rank + 1 - kk) % world_size + world_size) % world_size;
      recv_c = ((rank - kk) % world_size + world_size) % world_size;
    }
    for (int p = 0; p < pieces; ++p) {
      const int slot = k * pieces + p;
      const int64_t off = static_cast<int64_t>(send_c) * shard_elems + p * piece_elems;
      const void* src =
          k == 0 ? static_cast<const void*>(input + off) : static_cast<const void*>(output + off);

      // The copy stream carries nothing but back-to-back engine work: an SM
      // kernel between two memcpys stalls the engine for a launch round trip.
      if (k > 0) {
        FI_CE_CHECK(cudaStreamWaitEvent(ce.copy_stream, ce.add_done[p]));
      }
      FI_CE_CHECK(cudaMemcpyAsync(scratch_at(next, k, p), src, piece_bytes,
                                  cudaMemcpyDeviceToDevice, ce.copy_stream));
      FI_CE_CHECK(cudaEventRecord(ce.copied[slot], ce.copy_stream));

      FI_CE_CHECK(cudaStreamWaitEvent(ce.flag_stream, ce.copied[slot]));
      ce_publish_flag_kernel<<<1, 1, 0, ce.flag_stream>>>(flag_at(next, slot),
                                                          views.ce_send_counters + slot);
      ce_wait_flag_kernel<<<1, 1, 0, stream>>>(flag_at(rank, slot), views.ce_wait_counters + slot);

      const int64_t roff = static_cast<int64_t>(recv_c) * shard_elems + p * piece_elems;
      auto* landed = scratch_at(rank, k, p);
      if (reduce_phase) {
        ce_add_kernel<T><<<add_grid, threads, 0, stream>>>(
            reinterpret_cast<uint4*>(output + roff), reinterpret_cast<const uint4*>(input + roff),
            reinterpret_cast<const uint4*>(landed), packs);
      } else {
        FI_CE_CHECK(
            cudaMemcpyAsync(output + roff, landed, piece_bytes, cudaMemcpyDeviceToDevice, stream));
      }
      FI_CE_CHECK(cudaEventRecord(ce.add_done[p], stream));
    }
  }

  // Join. Mandatory: without it cudaStreamEndCapture fails with
  // cudaErrorStreamCaptureUnjoined, and in eager mode the caller reads a torn
  // result because the side streams are still running.
  FI_CE_CHECK(cudaEventRecord(ce.copy_done, ce.copy_stream));
  FI_CE_CHECK(cudaEventRecord(ce.flag_done, ce.flag_stream));
  FI_CE_CHECK(cudaStreamWaitEvent(stream, ce.copy_done));
  FI_CE_CHECK(cudaStreamWaitEvent(stream, ce.flag_done));

  // End-of-call rendezvous with the rank whose staging this one writes.
  //
  // It looks like dead weight, and deleting it leaves every test passing -- that
  // was measured, see _skewed_ranks_worker. The reason is that the current
  // gating makes the skew it closes unreachable: a step-k copy waits on the
  // step-(k-1) add, so `r@k` implies `(r-1)@(k-1)`. That is a property of the
  // `cudaStreamWaitEvent(copy_stream, add_done[p])` above, not of the ring --
  // relax that to let the copy engine run further ahead and the induction
  // collapses with nothing to report it.
  //
  // Whatever replaces it must still rendezvous with *every* consumer of this
  // rank's staging, which is why the island schedule below needs two slots and
  // this one needs one.
  const int hs = ce_handshake_slot(world_size);
  ce_publish_flag_kernel<<<1, 1, 0, stream>>>(flag_at(prev, hs), views.ce_send_counters + hs);
  ce_wait_flag_kernel<<<1, 1, 0, stream>>>(flag_at(rank, hs), views.ce_wait_counters + hs);
  return cudaGetLastError();
}

// 4+4 island decomposition: intra-island ring reduce-scatter, one cross-socket
// owner-pair exchange, intra-island ring all-gather. World size 8 only.
//
// Why it exists: in the flat ring every link carries the same 2(N-1)/N of the
// payload, so on a box whose eight GPUs sit in two NUMA islands the two hops
// that span the sockets become the critical path. Splitting the schedule puts
// one chunk per rank across that boundary instead of the full ring traffic,
// which is what buys it its edge over the flat ring there.
//
// And why it must be gated: the grouping below is `rank / 4` and `rank ^ 4`,
// which describes one topology -- two islands with a single crossing -- and
// nothing else. On a switch-paired fabric, where a NUMA node holds two PCIe
// switches with two GPUs behind each and there are three cost levels rather
// than two, it cuts across the hierarchy it exists to exploit and loses to both
// the flat ring and the SM path. Deriving the grouping from the topology probe
// instead of hardcoding it is the real fix and is not done; until then the
// caller must only select this on a fabric it describes.
template <typename T>
cudaError_t ce_ring_all_reduce_island(const T* input, T* output, int64_t numel,
                                      const PeerViews& views, int rank, int world_size,
                                      const WorkspaceLayout& layout, const CeResources& ce,
                                      int pieces_hint, int threads, cudaStream_t stream) {
  if (world_size != 8) return cudaErrorInvalidValue;
  constexpr int kPackElems = PackTraits<T>::kPackElems;
  constexpr int kIsland = 4;
  const int64_t chunk_elems = numel / kIsland;
  const size_t chunk_bytes = static_cast<size_t>(chunk_elems) * sizeof(T);
  const int pieces = ce_pick_pieces(chunk_elems, chunk_bytes, kPackElems, pieces_hint);
  const int64_t piece_elems = chunk_elems / pieces;
  const size_t piece_bytes = static_cast<size_t>(piece_elems) * sizeof(T);
  // Slots here stage a quarter of the payload, not an Nth.
  const size_t slot_stride = (layout.max_payload_bytes / 4 + 127u) & ~static_cast<size_t>(127u);

  const int island = rank / kIsland;
  const int local = rank % kIsland;
  const int next = island * kIsland + (local + 1) % kIsland;
  const int partner = rank ^ kIsland;

  const int64_t packs = piece_elems / kPackElems;
  const unsigned grid =
      static_cast<unsigned>(packs < threads ? 1 : (packs + threads - 1) / threads);
  const unsigned add_grid = grid > 64u ? 64u : grid;

  auto flag_at = [&](int peer, int slot) {
    return reinterpret_cast<int32_t*>(views.ce_flags[peer] +
                                      static_cast<uint64_t>(slot) * kCeFlagStride);
  };
  auto scratch_at = [&](int peer, int step, int p) {
    return reinterpret_cast<char*>(views.ce_scratch[peer]) +
           static_cast<ptrdiff_t>(step) * static_cast<ptrdiff_t>(slot_stride) +
           static_cast<ptrdiff_t>(p) * static_cast<ptrdiff_t>(piece_bytes);
  };
  auto in_at = [&](int c, int p) {
    return input + static_cast<int64_t>(c) * chunk_elems + p * piece_elems;
  };
  auto out_at = [&](int c, int p) {
    return output + static_cast<int64_t>(c) * chunk_elems + p * piece_elems;
  };

  FI_CE_CHECK(cudaEventRecord(ce.input_ready, stream));
  FI_CE_CHECK(cudaStreamWaitEvent(ce.copy_stream, ce.input_ready));
  FI_CE_CHECK(cudaStreamWaitEvent(ce.flag_stream, ce.input_ready));

  auto hop = [&](int slot, int peer, char* dst, const void* src,
                 const cudaEvent_t* gate) -> cudaError_t {
    if (gate != nullptr) FI_CE_CHECK(cudaStreamWaitEvent(ce.copy_stream, *gate));
    FI_CE_CHECK(cudaMemcpyAsync(dst, src, piece_bytes, cudaMemcpyDeviceToDevice, ce.copy_stream));
    FI_CE_CHECK(cudaEventRecord(ce.copied[slot], ce.copy_stream));
    FI_CE_CHECK(cudaStreamWaitEvent(ce.flag_stream, ce.copied[slot]));
    ce_publish_flag_kernel<<<1, 1, 0, ce.flag_stream>>>(flag_at(peer, slot),
                                                        views.ce_send_counters + slot);
    ce_wait_flag_kernel<<<1, 1, 0, stream>>>(flag_at(rank, slot), views.ce_wait_counters + slot);
    return cudaSuccess;
  };

  // After these three steps this rank owns chunk (local+1)%4, summed over its
  // own island.
  for (int k = 0; k < 3; ++k) {
    const int send_c = (local - k + 8) % kIsland;
    const int recv_c = (local - k - 1 + 8) % kIsland;
    for (int p = 0; p < pieces; ++p) {
      const int slot = k * pieces + p;
      const void* src = k == 0 ? static_cast<const void*>(in_at(send_c, p))
                               : static_cast<const void*>(out_at(send_c, p));
      FI_CE_CHECK(hop(slot, next, scratch_at(next, k, p), src, k == 0 ? nullptr : &ce.add_done[p]));
      ce_add_kernel<T><<<add_grid, threads, 0, stream>>>(
          reinterpret_cast<uint4*>(out_at(recv_c, p)),
          reinterpret_cast<const uint4*>(in_at(recv_c, p)),
          reinterpret_cast<const uint4*>(scratch_at(rank, k, p)), packs);
      FI_CE_CHECK(cudaEventRecord(ce.add_done[p], stream));
    }
  }

  // Phase B: the only cross-socket hop, one chunk each way.
  //
  // This step is the one place in either schedule where the send source and the
  // accumulate destination are the same buffer. ce_wait_flag orders this rank
  // against the *partner's* copy, not against its own outbound read, so without
  // the extra wait below the copy stream is still reading out[own] while the
  // add overwrites it and the partner receives a half-written chunk. The flat
  // ring cannot hit this -- send_chunk != recv_chunk always holds there -- so
  // cloning its structure walks straight into it.
  const int own = (local + 1) % kIsland;
  for (int p = 0; p < pieces; ++p) {
    const int slot = 3 * pieces + p;
    FI_CE_CHECK(hop(slot, partner, scratch_at(partner, 3, p), out_at(own, p), &ce.add_done[p]));
    FI_CE_CHECK(cudaStreamWaitEvent(stream, ce.copied[slot]));
    ce_add_kernel<T><<<add_grid, threads, 0, stream>>>(
        reinterpret_cast<uint4*>(out_at(own, p)), reinterpret_cast<const uint4*>(out_at(own, p)),
        reinterpret_cast<const uint4*>(scratch_at(rank, 3, p)), packs);
    FI_CE_CHECK(cudaEventRecord(ce.add_done[p], stream));
  }

  for (int k = 0; k < 3; ++k) {
    const int send_c = (local + 1 - k + 8) % kIsland;
    const int recv_c = (local - k + 8) % kIsland;
    for (int p = 0; p < pieces; ++p) {
      const int slot = (4 + k) * pieces + p;
      FI_CE_CHECK(hop(slot, next, scratch_at(next, 4 + k, p), out_at(send_c, p), &ce.add_done[p]));
      FI_CE_CHECK(cudaMemcpyAsync(out_at(recv_c, p), scratch_at(rank, 4 + k, p), piece_bytes,
                                  cudaMemcpyDeviceToDevice, stream));
      FI_CE_CHECK(cudaEventRecord(ce.add_done[p], stream));
    }
  }

  FI_CE_CHECK(cudaEventRecord(ce.copy_done, ce.copy_stream));
  FI_CE_CHECK(cudaEventRecord(ce.flag_done, ce.flag_stream));
  FI_CE_CHECK(cudaStreamWaitEvent(stream, ce.copy_done));
  FI_CE_CHECK(cudaStreamWaitEvent(stream, ce.flag_done));

  // Rendezvous with both consumers of this rank's staging: `island_next`, which
  // reads what phases A and C wrote, and `partner`, which reads phase B. One
  // global slot -- what the flat ring uses, where the only consumer is `next` --
  // leaves at least one of them uncovered on every rank here, and neither on the
  // two ranks that wrap within their island.
  const int island_prev = island * kIsland + (local - 1 + kIsland) % kIsland;
  const int hs = ce_handshake_slot(world_size);
  const int hs_partner = ce_partner_handshake_slot(world_size);
  ce_publish_flag_kernel<<<1, 1, 0, stream>>>(flag_at(island_prev, hs),
                                              views.ce_send_counters + hs);
  ce_publish_flag_kernel<<<1, 1, 0, stream>>>(flag_at(partner, hs_partner),
                                              views.ce_send_counters + hs_partner);
  ce_wait_flag_kernel<<<1, 1, 0, stream>>>(flag_at(rank, hs), views.ce_wait_counters + hs);
  ce_wait_flag_kernel<<<1, 1, 0, stream>>>(flag_at(rank, hs_partner),
                                           views.ce_wait_counters + hs_partner);
  return cudaGetLastError();
}

}  // namespace pcie_ipc
}  // namespace comm
}  // namespace flashinfer

#undef FI_CE_CHECK

#endif  // FLASHINFER_COMM_PCIE_IPC_CE_RING_CUH_
