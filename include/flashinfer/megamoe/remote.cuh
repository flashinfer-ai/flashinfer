#pragma once

// Inter-node links of the SM90 MegaMoE kernels. A rank talks to one mirror rank
// on every node it cannot reach over NVLink: it puts packed token rows into the
// mirror's inbox and receives reduced rows into its return inbox
// (layout::RemoteRegions). Two carriers implement the same put / signal / wait
// contract: load/store through a peer pointer when the mirror is on this host
// (how the protocol is tested on one node), and NCCL GIN puts with signals on
// the rank's symmetric window otherwise. The GIN path is compiled only with
// FI_MEGAMOE_WITH_NCCL_GIN, so single-node builds need no NCCL headers.

#include <flashinfer/megamoe/detail/common.cuh>
#include <flashinfer/megamoe/detail/ptx.cuh>

#if defined(FI_MEGAMOE_WITH_NCCL_GIN)
#ifndef _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE
#define _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE 1
#endif
#include <nccl_device.h>
#endif

namespace flashinfer::megamoe::comm {

// How this rank reaches its mirror on one remote node.
struct RemotePeer {
  // World rank of the mirror.
  uint32_t rank;
  // Index of this rank's node among the mirror's remote nodes: which inbox of
  // the mirror receives this rank's tokens.
  uint32_t node_at_mirror;
  // Byte offset from this rank's buffer to the mirror's when the mirror is
  // load/store reachable; 0 selects the GIN carrier.
  int64_t lsa_offset;
};

template <uint32_t kNumRemoteNodes>
struct RemoteLinks {
  RemotePeer peers[kNumRemoteNodes > 0 ? kNumRemoteNodes : 1];
  // GIN carrier: device copy of the ncclDevComm, the window handle of this
  // rank's buffer (symmetric, so the same handle names the peer's window) and
  // the byte address the window starts at on this rank.
  const void* dev_comm;
  void* window;
  int64_t window_base;
  uint32_t gin_context;
};

// Signal ids: two per remote node, inbox then return.
CUTLASS_HOST_DEVICE constexpr uint32_t remote_inbox_signal(const uint32_t& node) {
  return node * 2;
}
CUTLASS_HOST_DEVICE constexpr uint32_t remote_return_signal(const uint32_t& node) {
  return node * 2 + 1;
}

#if defined(__CUDA_ARCH__) or defined(__CLION_IDE__)

// Warp-cooperative copy of `bytes` (multiple of 16) from `src` to `dst`: up to
// eight 16-byte loads per lane are issued before their stores, so a 4 KB row
// costs one memory latency instead of eight.
CUTLASS_DEVICE void copy_16b_warp(void* dst, const void* src, const uint64_t& bytes,
                                  const uint32_t& lane_idx) {
  auto* __restrict__ d = static_cast<uint4*>(dst);
  const auto* __restrict__ s = static_cast<const uint4*>(src);
  const uint64_t n = bytes / sizeof(uint4);
  for (uint64_t base = 0; base < n; base += 8 * 32) {
    uint4 v[8];
#pragma unroll
    for (uint32_t u = 0; u < 8; ++u) {
      const uint64_t i = base + u * 32 + lane_idx;
      if (i < n) v[u] = s[i];
    }
#pragma unroll
    for (uint32_t u = 0; u < 8; ++u) {
      const uint64_t i = base + u * 32 + lane_idx;
      if (i < n) d[i] = v[u];
    }
  }
}

template <uint32_t kNumRemoteNodes>
struct RemoteIO {
  const RemoteLinks<kNumRemoteNodes>& links;

  CUTLASS_DEVICE explicit RemoteIO(const RemoteLinks<kNumRemoteNodes>& links) : links(links) {}

  CUTLASS_DEVICE bool is_lsa(const uint32_t& node) const {
    return links.peers[node].lsa_offset != 0;
  }

  // Put `bytes` from `src` (inside this rank's buffer) to the mirror on `node`,
  // at the address `dst_equiv` has in this rank's own layout. Called by a full
  // warp. With `signal`, the mirror's `flag_equiv` word (load/store carrier) or
  // the GIN signal `signal_id` is incremented once the data is visible; a put
  // without a signal is ordered before a later signalled put to the same node.
  CUTLASS_DEVICE void put_warp(const uint32_t& node, void* dst_equiv, const void* src,
                               const uint64_t& bytes, const uint32_t& lane_idx, const bool& signal,
                               uint64_t* flag_equiv, const uint32_t& signal_id) const {
    const auto& peer = links.peers[node];
    if (is_lsa(node)) {
      auto* dst = reinterpret_cast<void*>(reinterpret_cast<int64_t>(dst_equiv) + peer.lsa_offset);
      copy_16b_warp(dst, src, bytes, lane_idx);
      if (signal) {
        __threadfence_system();
        __syncwarp();
        if (lane_idx == 0)
          ptx::atomic_add_sys(reinterpret_cast<uint64_t*>(reinterpret_cast<int64_t>(flag_equiv) +
                                                          peer.lsa_offset),
                              1ull);
      }
      __syncwarp();
      return;
    }
#if defined(FI_MEGAMOE_WITH_NCCL_GIN)
    if (lane_idx == 0) {
      const auto* dev_comm = static_cast<const ncclDevComm*>(links.dev_comm);
      ncclGin gin{*dev_comm, static_cast<int>(links.gin_context)};
      const auto window = reinterpret_cast<ncclWindow_t>(links.window);
      const auto dst_off = static_cast<size_t>(reinterpret_cast<int64_t>(dst_equiv) - links.window_base);
      const auto src_off = static_cast<size_t>(reinterpret_cast<int64_t>(src) - links.window_base);
      if (signal)
        gin.put(ncclTeamWorld(*dev_comm), static_cast<int>(peer.rank), window, dst_off, window,
                src_off, bytes, ncclGin_StrongSignalAdd{static_cast<ncclGinSignal_t>(signal_id), 1ull});
      else
        gin.put(ncclTeamWorld(*dev_comm), static_cast<int>(peer.rank), window, dst_off, window,
                src_off, bytes);
    }
    __syncwarp();
#else
    FI_MEGAMOE_DEVICE_ASSERT(false and "GIN carrier not compiled in");
#endif
  }

  // Wait until the mirror on `node` has signalled at least `least` times
  // (load/store carrier: the local `flag` word; GIN: `signal_id`). One thread.
  CUTLASS_DEVICE void wait(const uint32_t& node, const uint64_t* flag, const uint32_t& signal_id,
                           const uint64_t& least) const {
    if (is_lsa(node)) {
      while (ptx::ld_acq_sys(flag) < least) {
      }
      return;
    }
#if defined(FI_MEGAMOE_WITH_NCCL_GIN)
    const auto* dev_comm = static_cast<const ncclDevComm*>(links.dev_comm);
    ncclGin gin{*dev_comm, static_cast<int>(links.gin_context)};
    gin.waitSignal(ncclCoopThread{}, static_cast<ncclGinSignal_t>(signal_id), least);
#else
    FI_MEGAMOE_DEVICE_ASSERT(false and "GIN carrier not compiled in");
#endif
  }
};

#endif

}  // namespace flashinfer::megamoe::comm
